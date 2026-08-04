"""
Unit test suite for chakra terminal UI components and REPL prompt handling.
"""

from io import StringIO
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch

from chakra.agent import KimiAgent
from chakra.cli import run_repl
from chakra.model import K3Model, tiny_config
from chakra.tokenizer import KimiTokenizer
from chakra import ui


class TestTerminalUI(unittest.TestCase):
    """Test suite for chakra.ui module functions and formatting."""

    def test_code_box_drawing(self) -> None:
        captured_output = StringIO()
        sample_code = "def hello():\n    print('Hello World!')"
        with patch("sys.stdout", captured_output):
            ui.print_code_box(sample_code, title="TEST CODE")

        output = captured_output.getvalue()
        self.assertIn("TEST CODE", output)
        self.assertIn("def hello():", output)
        self.assertIn("print('Hello World!')", output)
        self.assertIn("┌─", output)
        self.assertIn("└", output)

    def test_code_box_empty_code(self) -> None:
        captured_output = StringIO()
        with patch("sys.stdout", captured_output):
            ui.print_code_box("", title="EMPTY BOX")

        output = captured_output.getvalue()
        self.assertIn("EMPTY BOX", output)
        self.assertIn("# No code generated", output)

    def test_step_formatting(self) -> None:
        statuses = ["INFO", "SUCCESS", "FAIL", "WARN", "WAIT", "UNKNOWN"]
        for st in statuses:
            captured_output = StringIO()
            with patch("sys.stdout", captured_output):
                ui.print_step("TEST_STEP", f"Description for {st}", status=st)

            output = captured_output.getvalue()
            self.assertIn("⟡ [TEST_STEP]", output)
            self.assertIn(f"Description for {st}", output)
            if st == "WAIT":
                self.assertIn("[RUNNING]", output)
            elif st == "UNKNOWN":
                self.assertIn("[UNKNOWN]", output)
            else:
                self.assertIn(f"[{st}]", output)

    def test_step_formatting_no_description(self) -> None:
        captured_output = StringIO()
        with patch("sys.stdout", captured_output):
            ui.print_step("SOLO_STEP", status="SUCCESS")

        output = captured_output.getvalue()
        self.assertIn("⟡ [SOLO_STEP]", output)
        self.assertIn("[SUCCESS]", output)

    def test_screen_clearing(self) -> None:
        with patch("os.system") as mock_system:
            ui.clear_screen()
            expected_cmd = "cls" if os.name == "nt" else "clear"
            mock_system.assert_called_once_with(expected_cmd)

    def test_print_banner(self) -> None:
        captured_output = StringIO()
        with patch("sys.stdout", captured_output):
            ui.print_banner()

        output = captured_output.getvalue()
        self.assertIn("CHAKRA-AI TERMINAL", output)
        self.assertIn("Chakra-AI Engine", output)


class TestREPLPromptHandling(unittest.TestCase):
    """Test suite for REPL shell input prompt handling."""

    def setUp(self) -> None:
        cfg = tiny_config()
        self.model = K3Model(cfg)
        self.model.eval()
        self.tokenizer = KimiTokenizer(mode="auto")
        self.agent = KimiAgent(model=self.model, tokenizer=self.tokenizer, device="cpu")

    def test_repl_direct_prompt_without_code(self) -> None:
        user_inputs = ["Write hello world script", "/exit"]
        captured_output = StringIO()

        mock_agent = MagicMock()
        mock_agent.run_agentic_loop.return_value = {
            "success": True,
            "iterations": 1,
            "code": "print('Hello World')",
            "stdout": "Hello World",
            "stderr": "",
        }

        with patch("builtins.input", side_effect=user_inputs), \
             patch("sys.stdout", captured_output):
            run_repl(
                model=self.model,
                tokenizer=self.tokenizer,
                agent=mock_agent,
                gen_tokens=4,
                device="cpu",
                incremental=False,
            )

        output = captured_output.getvalue()
        self.assertIn("Chakra-AI Agentic Terminal Active", output)
        self.assertIn("Running task: 'Write hello world script'", output)
        self.assertIn("Code saved to", output)
        self.assertIn("Task completed successfully", output)
        self.assertIn("Exiting Chakra-AI", output)

        mock_agent.run_agentic_loop.assert_called_once()
        call_kwargs = mock_agent.run_agentic_loop.call_args.kwargs
        self.assertIn("Write hello world script", call_kwargs["prompt"])

    def test_repl_code_prefix_handling(self) -> None:
        user_inputs = ["/code Write fibonacci", "/exit"]
        captured_output = StringIO()

        mock_agent = MagicMock()
        mock_agent.run_agentic_loop.return_value = {
            "success": True,
            "iterations": 1,
            "code": "def fib(n): pass",
            "stdout": "Done",
            "stderr": "",
        }

        with patch("builtins.input", side_effect=user_inputs), \
             patch("sys.stdout", captured_output):
            run_repl(
                model=self.model,
                tokenizer=self.tokenizer,
                agent=mock_agent,
                gen_tokens=4,
                device="cpu",
                incremental=False,
            )

        mock_agent.run_agentic_loop.assert_called_once()
        call_kwargs = mock_agent.run_agentic_loop.call_args.kwargs
        self.assertIn("Write fibonacci", call_kwargs["prompt"])

    def test_repl_direct_prompt_failure_handling(self) -> None:
        user_inputs = ["Broken task", "quit"]
        captured_output = StringIO()

        mock_agent = MagicMock()
        mock_agent.run_agentic_loop.return_value = {
            "success": False,
            "iterations": 3,
            "code": "raise Error()",
            "stdout": "",
            "stderr": "RuntimeError: custom error",
        }

        with patch("builtins.input", side_effect=user_inputs), \
             patch("sys.stdout", captured_output):
            run_repl(
                model=self.model,
                tokenizer=self.tokenizer,
                agent=mock_agent,
                gen_tokens=4,
                device="cpu",
                incremental=False,
            )

        output = captured_output.getvalue()
        self.assertIn("Running task: 'Broken task'", output)
        self.assertIn("Code saved to", output)


if __name__ == "__main__":
    unittest.main()
