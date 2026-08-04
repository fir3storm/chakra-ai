# tests/unit/test_cli_ui.py
"""
Unit tests for chakra.ui helpers and chakra.cli REPL redesign.
"""
import io
import sys
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from chakra.ui import print_banner, print_code_box, print_step
from chakra.cli import run_repl


class TestKimipyUI(unittest.TestCase):
    def test_print_banner(self):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            print_banner()
        out = buf.getvalue()
        self.assertIn("CHAKRA-AI TERMINAL", out)
        self.assertIn("Abhirup Guha", out)

    def test_print_step(self):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            print_step("AGENT", "Testing agent step", "SUCCESS")
        out = buf.getvalue()
        self.assertIn("AGENT", out)
        self.assertIn("Testing agent step", out)

    def test_print_code_box(self):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            print_code_box("print('hello world')", title="TEST CODE")
        out = buf.getvalue()
        self.assertIn("TEST CODE", out)
        self.assertIn("print('hello world')", out)

    def test_run_repl_natural_prompt_and_exit(self):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_agent = MagicMock()
        mock_agent.last_code = "print('hello')"
        mock_agent.run_agentic_loop.return_value = {
            "success": True,
            "code": "print('hello')",
            "stdout": "hello\n",
            "stderr": "",
            "iterations": 1,
        }

        # Test user entering natural prompt followed by /exit
        user_inputs = ["Write a print hello script", "/exit"]
        buf = io.StringIO()

        with patch("builtins.input", side_effect=user_inputs) as mock_input, patch("sys.stdout", buf):
            run_repl(
                model=mock_model,
                tokenizer=mock_tokenizer,
                agent=mock_agent,
                gen_tokens=16,
                device="cpu",
                incremental=True,
            )

        out = buf.getvalue()
        self.assertEqual(mock_input.call_args[0][0], "(chakra-ai) > ")
        self.assertIn("Exiting Chakra-AI Agentic Terminal", out)
        mock_agent.run_agentic_loop.assert_called_once()
        call_kwargs = mock_agent.run_agentic_loop.call_args.kwargs
        self.assertIn("Write a print hello script", call_kwargs["prompt"])
        self.assertEqual(call_kwargs["max_retries"], 3)
        self.assertEqual(call_kwargs["gen_tokens"], 16)
        self.assertEqual(call_kwargs["incremental"], True)

    def test_run_repl_shortcuts(self):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_agent = MagicMock()
        mock_agent.last_code = "x = 42"

        user_inputs = ["/help", "/clear", "/exit"]
        buf = io.StringIO()

        with patch("builtins.input", side_effect=user_inputs), patch("sys.stdout", buf):
            run_repl(
                model=mock_model,
                tokenizer=mock_tokenizer,
                agent=mock_agent,
                gen_tokens=16,
                device="cpu",
                incremental=True,
            )

        out = buf.getvalue()
        self.assertIn("Chakra-AI Agentic REPL Shell Commands:", out)


if __name__ == "__main__":
    unittest.main()
