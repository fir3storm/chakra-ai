# tests/test_new_enhancements.py
"""
Unit tests for Chakra AI enhancements:
- Security auto-remediation (/fix-sec)
- Dependency & Dockerfile security auditing
- Multi-language sandbox execution
- Automated test and fix loop (/test)
- AST Symbol Graph (/ast)
- CLI --audit and --lsp flags
"""

import sys
import tempfile
from pathlib import Path
import pytest

from chakra.security import InfoSecAuditor
from chakra.agent import KimiAgent
from chakra.workspace import WorkspaceIndexer, SymbolGraph, scan_ast_symbols
from chakra.cli import main


def test_security_auto_remediation():
    auditor = InfoSecAuditor()
    vulnerable_code = (
        "import os\n"
        "SECRET_KEY = 'supersecret123'\n"
        "eval('2 + 2')\n"
        "os.system('dir')\n"
    )
    audit_res = auditor.audit_code(vulnerable_code)
    vulns = audit_res.get("vulnerabilities", [])
    assert len(vulns) > 0

    remediated_code, fixes_applied = auditor.fix_vulnerabilities(vulnerable_code, vulns)
    assert fixes_applied > 0
    assert "os.getenv('SECRET_KEY'" in remediated_code or "ast.literal_eval" in remediated_code or "subprocess.run" in remediated_code


def test_auto_remediate_file(tmp_path):
    auditor = InfoSecAuditor()
    file_path = tmp_path / "vulnerable_script.py"
    file_path.write_text("API_TOKEN = '1234567890abcdef'\n", encoding="utf-8")

    res = auditor.auto_remediate_file(file_path)
    assert res["fixes_applied"] > 0
    updated_content = file_path.read_text(encoding="utf-8")
    assert "os.getenv('API_TOKEN'" in updated_content


def test_dependency_and_dockerfile_auditing(tmp_path):
    auditor = InfoSecAuditor()

    # Create dummy requirements.txt and Dockerfile
    req_file = tmp_path / "requirements.txt"
    req_file.write_text("flask\npickle\n", encoding="utf-8")

    docker_file = tmp_path / "Dockerfile"
    docker_file.write_text("FROM python:latest\nENV SECRET=xyz\nRUN curl http://example.com | sh\n", encoding="utf-8")

    dep_findings = auditor.audit_dependencies(tmp_path)
    assert len(dep_findings) >= 2  # unpinned flask, vulnerable pickle

    docker_findings = auditor.audit_dockerfile(docker_file)
    assert len(docker_findings) >= 3  # latest tag, secret in env, piped curl


def test_multi_language_sandbox_execution():
    agent = KimiAgent()

    # Python execution
    py_res = agent.execute_sandbox("print('Hello Python')", language="python")
    assert py_res["success"] is True
    assert "Hello Python" in py_res["stdout"]

    # JS auto-detect fallback if node available
    js_code = "console.log('Hello JS');"
    js_res = agent.execute_sandbox(js_code, language="javascript")
    assert "returncode" in js_res


def test_automated_test_and_fix_loop(tmp_path):
    agent = KimiAgent()

    test_file = tmp_path / "test_dummy.py"
    test_file.write_text("def test_pass(): assert 1 == 1\nif __name__ == '__main__': test_pass()\n", encoding="utf-8")

    res = agent.test_and_fix(test_cmd=[sys.executable, str(test_file)], max_attempts=1)
    assert res["success"] is True
    assert res["attempts"] == 1


def test_ast_symbol_graph(tmp_path):
    py_file = tmp_path / "sample.py"
    py_file.write_text(
        "def hello(name: str) -> str:\n"
        "    return f'Hello {name}'\n\n"
        "class Calculator:\n"
        "    def add(self, a, b):\n"
        "        return a + b\n",
        encoding="utf-8",
    )

    indexer = WorkspaceIndexer(root_dir=tmp_path)
    graph = scan_ast_symbols(indexer)
    summary = graph.summary()

    assert "hello" in summary
    assert "Calculator" in summary


def test_cli_flags(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    # Test --lsp flag
    ret_lsp = main(["--lsp"])
    assert ret_lsp == 0

    # Test --audit flag
    py_file = tmp_path / "clean_script.py"
    py_file.write_text("print('clean')", encoding="utf-8")

    ret_audit = main(["--audit"])
    assert ret_audit == 0


def test_turbo_llama_backend():
    from chakra.engine_llama import LlamaCppBackend
    backend = LlamaCppBackend(enable_turbo=True)
    if backend.loaded:
        assert backend.enable_turbo is True
        grammar = backend.get_python_grammar()
        assert grammar is not None


def test_grammar_constrained_choice_and_json():
    """generate_choice/generate_json back the tool-calling loop in chakra.tools.ToolLoop —
    when a real GGUF model is loaded, output must be grammar-valid every time."""
    from chakra.engine_llama import LlamaCppBackend
    backend = LlamaCppBackend()
    if not backend.loaded:
        return

    choice = backend.generate_choice(prompt="Pick one.", choices=["CALL_TOOL", "FINAL_ANSWER"])
    assert choice in ("CALL_TOOL", "FINAL_ANSWER")

    schema = {
        "type": "object",
        "properties": {"tool": {"enum": ["read_file", "write_file"]}, "args": {"type": "object"}},
        "required": ["tool", "args"],
    }
    result = backend.generate_json(prompt="Call read_file on agent.py", schema=schema)
    assert result is not None
    assert result["tool"] in ("read_file", "write_file")
    assert isinstance(result["args"], dict)

    # Empty choices / no schema should fail closed, not raise.
    assert backend.generate_choice(prompt="x", choices=[]) == ""
    assert backend.generate_json(prompt="x", schema=None) is None


def test_tool_registry_and_search_replace():
    from chakra.agent import ToolRegistry, apply_search_replace_block

    # Test Search & Replace block editing
    orig_code = "def foo():\n    print('old')\n    return 42\n"
    search_b = "    print('old')"
    replace_b = "    print('new')"
    new_code, success = apply_search_replace_block(orig_code, search_b, replace_b)
    assert success is True
    assert "print('new')" in new_code

    # Test ToolRegistry
    reg = ToolRegistry()
    reg.register("add", lambda a, b: a + b, "Add numbers")

    res = reg.execute("add", a=5, b=10)
    assert res["success"] is True
    assert res["result"] == 15

def test_auto_continuation_and_progress_bar():
    from chakra.agent import KimiAgent
    from chakra.ui import ProgressBar

    # Test ProgressBar
    pbar = ProgressBar(title="Test Task", total_passes=2)
    pbar.update(current_pass=1, new_tokens=50, status="Testing")
    pbar.finish(message="Done")

    # Test auto-continuation logic with dummy model
    class TruncatedModel:
        def __init__(self):
            self.calls = 0
        def __call__(self, prompt):
            self.calls += 1
            if self.calls == 1:
                return "```python\ndef my_func():\n    x = 10"
            else:
                return "    return x * 2\n```"

        def generate(self, prompt, n_new=512, system=""):
            return self.__call__(prompt)

    agent = KimiAgent(model=TruncatedModel())
    full_code = agent.generate_with_auto_continuation("Write my_func", max_passes=3)
    assert "def my_func():" in full_code
    assert "return x * 2" in full_code
    assert full_code.count("```") == 2



