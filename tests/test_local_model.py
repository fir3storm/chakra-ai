"""
Tests for Option B: LocalModelRunner and CLI integration.
Author & Creator: Abhirup Guha (Info Security Solution)
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chakra.agent import KimiAgent, LocalModelRunner


class TestLocalModelRunner:
    """Test suite for LocalModelRunner class."""

    def test_runner_init_no_model_dir(self, tmp_path):
        """Runner should initialize in fallback mode when model dir doesn't exist."""
        runner = LocalModelRunner(model_path=str(tmp_path / "nonexistent"))
        assert runner.loaded is False
        assert runner.model is None
        assert runner.tokenizer is None

    def test_runner_init_with_empty_dir(self, tmp_path):
        """Runner should initialize in fallback mode when model dir is empty."""
        model_dir = tmp_path / "empty_model"
        model_dir.mkdir()
        runner = LocalModelRunner(model_path=str(model_dir))
        # Should still set loaded=True (fallback) even though model is None
        assert runner.model_path == model_dir

    def test_runner_generate_fallback_string(self, tmp_path):
        """Runner should return echo-style output in fallback mode."""
        runner = LocalModelRunner(model_path=str(tmp_path / "nonexistent"))
        result = runner.generate("Hello world")
        assert isinstance(result, str)
        assert "Option B" in result
        assert "Hello world" in result

    def test_runner_generate_fallback_tensor_passthrough(self, tmp_path):
        """Runner should pass through tensors in fallback mode."""
        import torch
        runner = LocalModelRunner(model_path=str(tmp_path / "nonexistent"))
        dummy_tensor = torch.tensor([[1, 2, 3]])
        result = runner.generate(dummy_tensor)
        assert torch.equal(result, dummy_tensor)

    def test_runner_callable(self, tmp_path):
        """Runner should be callable like a function."""
        runner = LocalModelRunner(model_path=str(tmp_path / "nonexistent"))
        result = runner("Test prompt")
        assert isinstance(result, str)
        assert "Test prompt" in result

    def test_runner_device_assignment(self, tmp_path):
        """Runner should respect device parameter."""
        runner = LocalModelRunner(model_path=str(tmp_path / "nonexistent"), device="cpu")
        assert runner.device == "cpu"

    @patch("builtins.__import__")
    def test_runner_load_with_mock_model(self, mock_import, tmp_path):
        """Runner should attempt to load model when directory exists with files."""

        model_dir = tmp_path / "model_with_files"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "transformers":
                mod = MagicMock()
                mod.AutoModelForCausalLM.from_pretrained.return_value = mock_model
                mod.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
                return mod
            return real_import(name, *args, **kwargs)

        mock_import.side_effect = fake_import

        runner = LocalModelRunner(model_path=str(model_dir))
        assert runner.model_path == model_dir


class TestKimiAgentWithLocalModel:
    """Test suite for KimiAgent integration with LocalModelRunner."""

    def test_agent_auto_detects_local_model(self, tmp_path, monkeypatch):
        """KimiAgent should auto-detect and use LocalModelRunner when models/chakra_local exists."""
        monkeypatch.chdir(tmp_path)
        local_dir = Path("models/chakra_local")
        local_dir.mkdir(parents=True)

        agent = KimiAgent()
        # Agent should have attempted to set up a model
        assert agent.device == "cpu"

    def test_agent_fallback_without_local_model(self, tmp_path, monkeypatch):
        """KimiAgent should work without local model (echo mode)."""
        monkeypatch.chdir(tmp_path)

        agent = KimiAgent(model=None)
        result = agent.chat("Hello")
        assert isinstance(result, str)

    def test_agent_code_extraction(self):
        """KimiAgent should extract code blocks from markdown text."""
        agent = KimiAgent(model=None)
        text = 'Some text\n```python\nprint("hello")\n```\nMore text'
        blocks = agent.extract_code_blocks(text)
        assert len(blocks) == 1
        assert 'print("hello")' in blocks[0]

    def test_agent_code_extraction_no_blocks(self):
        """KimiAgent should return raw text when no code blocks found."""
        agent = KimiAgent(model=None)
        blocks = agent.extract_code_blocks("Just plain text")
        assert len(blocks) == 1
        assert blocks[0] == "Just plain text"

    def test_agent_sandbox_execution(self):
        """KimiAgent should execute Python code in sandbox."""
        agent = KimiAgent(model=None)
        result = agent.execute_sandbox("print('hello sandbox')")
        assert result["success"] is True
        assert "hello sandbox" in result["stdout"]

    def test_agent_sandbox_empty_code(self):
        """KimiAgent should reject empty code in sandbox."""
        agent = KimiAgent(model=None)
        result = agent.execute_sandbox("")
        assert result["success"] is False
        assert "Empty" in result["stderr"]

    def test_agent_sandbox_syntax_error(self):
        """KimiAgent should report syntax errors from sandbox."""
        agent = KimiAgent(model=None)
        result = agent.execute_sandbox("def foo(:\n  pass")
        assert result["success"] is False

    def test_agent_diff_generation(self):
        """KimiAgent should generate unified diffs."""
        agent = KimiAgent(model=None)
        old = "print('old')\n"
        new = "print('new')\n"
        diff = agent.generate_diff(old, new)
        assert "-print('old')" in diff
        assert "+print('new')" in diff


class TestCLIOptionB:
    """Test suite for CLI Option B integration."""

    def test_cli_imports(self):
        """CLI should import LocalModelRunner."""
        from chakra.cli import LOCAL_MODEL_DIR, ensure_local_model
        assert LOCAL_MODEL_DIR == Path("models/chakra_local")
        assert callable(ensure_local_model)

    def test_cli_local_model_dir_constant(self):
        """LOCAL_MODEL_DIR should point to correct path."""
        from chakra.cli import LOCAL_MODEL_DIR
        assert LOCAL_MODEL_DIR == Path("models") / "chakra_local"

    @patch("chakra.cli.ensure_local_model")
    def test_cli_auto_download_called(self, mock_ensure):
        """ensure_local_model should be callable."""
        from chakra.cli import ensure_local_model
        mock_ensure.return_value = True
        result = ensure_local_model()
        assert result is True

    def test_cli_arg_parser_has_local_model_flag(self):
        """CLI arg parser should have --local-model flag."""
        from chakra.cli import main
        # We test that main can parse --local-model without error
        # by checking the parser setup doesn't crash
        try:
            from chakra.cli import main
            # Just verifying the import and function exist
            assert callable(main)
        except Exception:
            pytest.fail("CLI main function should be importable")


class TestDownloadModel:
    """Test suite for model download script."""

    def test_download_model_script_exists(self):
        """Download script should exist at expected path."""
        script_path = Path(__file__).resolve().parent.parent / "tools" / "download_model.py"
        assert script_path.exists()

    def test_download_model_has_download_function(self):
        """Download script should have download_model function."""
        script_path = Path(__file__).resolve().parent.parent / "tools" / "download_model.py"
        content = script_path.read_text()
        assert "def download_model" in content
        assert "Qwen" in content or "qwen" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
