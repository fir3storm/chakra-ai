"""
Unit tests for KimiAgent in chakra.agent.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

from chakra.agent import KimiAgent


class TestKimiAgent(unittest.TestCase):
    def setUp(self):
        self.agent = KimiAgent()

    def test_extract_code_blocks(self):
        # Python fenced block
        text = "Here is the code:\n```python\nprint('hello world')\n```\nDone."
        blocks = self.agent.extract_code_blocks(text)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0], "print('hello world')")

        # Unspecified language block
        text_plain = "```\nx = 10\ny = 20\n```"
        blocks_plain = self.agent.extract_code_blocks(text_plain)
        self.assertEqual(blocks_plain, ["x = 10\ny = 20"])

        # Fallback raw text
        raw_text = "a = 1 + 1"
        blocks_raw = self.agent.extract_code_blocks(raw_text)
        self.assertEqual(blocks_raw, ["a = 1 + 1"])

    def test_run_in_sandbox_success(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("import sys\nsys.stdout.write('SUCCESS_OUT')\n")
            tmp_path = f.name

        try:
            res = self.agent.run_in_sandbox(tmp_path, timeout=5)
            self.assertTrue(res["success"])
            self.assertEqual(res["exit_code"], 0)
            self.assertEqual(res["stdout"], "SUCCESS_OUT")
            self.assertFalse(res["timed_out"])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_run_in_sandbox_failure(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("raise RuntimeError('Intentional Error')\n")
            tmp_path = f.name

        try:
            res = self.agent.run_in_sandbox(tmp_path, timeout=5)
            self.assertFalse(res["success"])
            self.assertNotEqual(res["exit_code"], 0)
            self.assertIn("Intentional Error", res["stderr"])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_run_in_sandbox_timeout(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("import time\ntime.sleep(10)\n")
            tmp_path = f.name

        try:
            res = self.agent.run_in_sandbox(tmp_path, timeout=1)
            self.assertFalse(res["success"])
            self.assertTrue(res["timed_out"])
            self.assertIn("timed out", res["stderr"])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_save_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "subdir" / "output.py"
            saved_path = self.agent.save_file(out_file, "print('saved')")
            self.assertTrue(os.path.isfile(saved_path))
            with open(saved_path, "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), "print('saved')")

    def test_self_debug_loop_recovery(self):
        # Mock model that fails on attempt 1 and fixes code on attempt 2
        attempts = 0

        def mock_model(prompt):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                # Buggy code on first attempt
                return "```python\nprint(undefined_variable)\n```"
            else:
                # Fixed code on retry
                return "```python\nprint('fixed variable')\n```"

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "fixed_script.py"
            res = self.agent.self_debug_loop(
                model=mock_model,
                tokenizer=None,
                task_prompt="Print fixed variable",
                max_retries=3,
                output_path=out_path,
            )

            self.assertTrue(res["success"])
            self.assertEqual(res["attempts"], 2)
            self.assertIn("fixed variable", res["code"])
            self.assertTrue(os.path.isfile(out_path))

    def test_offline_chat(self):
        def mock_model(prompt):
            return "Hello! I am ready."

        agent = KimiAgent(model=mock_model)
        reply = agent.chat("Hi Kimi")
        self.assertEqual(reply, "Hello! I am ready.")
        self.assertEqual(len(agent.chat_history), 2)
        self.assertEqual(agent.chat_history[0]["content"], "Hi Kimi")
        self.assertEqual(agent.chat_history[1]["content"], "Hello! I am ready.")

        agent.reset_chat()
        self.assertEqual(len(agent.chat_history), 0)


if __name__ == "__main__":
    unittest.main()
