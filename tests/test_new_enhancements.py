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
from chakra.cli import main, run_code_generation_task


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


def test_os_system_ternary_of_constants_not_flagged():
    """A hardcoded cross-platform screen-clear (`os.system("cls" if os.name == "nt" else
    "clear")`) is not attacker-influenceable and shouldn't be flagged as command injection —
    this is the standard idiom terminal animations use."""
    auditor = InfoSecAuditor()
    code = 'import os\nos.system("cls" if os.name == "nt" else "clear")\n'
    result = auditor.audit_code(code)
    cmd_vulns = [v for v in result.get("vulnerabilities", []) if v.get("rule_id") == "SEC-CMD-01"]
    assert cmd_vulns == []


def test_os_system_dynamic_arg_still_flagged():
    """A dynamic/variable command argument to os.system is still real command injection risk
    and must still be flagged — only the constant-ternary idiom got the exemption."""
    auditor = InfoSecAuditor()
    code = "import os\ndef run(cmd):\n    os.system(cmd)\n"
    result = auditor.audit_code(code)
    cmd_vulns = [v for v in result.get("vulnerabilities", []) if v.get("rule_id") == "SEC-CMD-01"]
    assert len(cmd_vulns) == 1


def test_is_long_running_script_detection():
    from chakra.agent import is_long_running_script

    animation = "import time\nwhile True:\n    print('.')\n    time.sleep(0.1)\n"
    assert is_long_running_script(animation) is True

    normal_script = "def add(a, b):\n    return a + b\nprint(add(1, 2))\n"
    assert is_long_running_script(normal_script) is False

    # A `while True:` with no sleep/curses/cls isn't confidently "intentional" — likely just a
    # bug, so it should NOT get the long-running exemption (still treated as a real failure).
    bare_infinite_loop = "while True:\n    x = 1\n"
    assert is_long_running_script(bare_infinite_loop) is False


def test_self_debug_loop_long_running_script_soft_succeeds():
    """A verified-working animation script (infinite loop + time.sleep) should be treated as a
    success on the first attempt once it survives the (shortened) sandbox timeout cleanly,
    instead of being retried up to max_retries as if it were broken."""
    import time as _time

    bird_code = (
        "```python\n"
        "import time\n"
        "def main():\n"
        "    x = 0\n"
        "    while True:\n"
        "        print('>=>' + ' ' * x)\n"
        "        x = (x + 1) % 10\n"
        "        time.sleep(0.1)\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
        "```"
    )

    class OneShotModel:
        def __call__(self, prompt):
            return bird_code

        def generate(self, prompt, n_new=512, system=""):
            return bird_code

    agent = KimiAgent(model=OneShotModel())
    t0 = _time.time()
    res = agent.run_agentic_loop("make an ascii animation that never stops", max_retries=3, gen_tokens=64)
    elapsed = _time.time() - t0

    assert res["success"] is True
    assert res["attempts"] == 1
    # Shortened (3s) timeout for detected long-running scripts, not the default 10s x 3 retries.
    assert elapsed < 8


def test_run_code_generation_task_does_not_reexecute_verified_code(tmp_path, monkeypatch):
    """run_code_generation_task must trust the success/stdout/stderr that run_agentic_loop
    already produced (self_debug_loop verifies GUI apps headlessly and long-running scripts
    correctly) rather than re-running the real saved file a second time — re-running would
    launch a real Tkinter window and block for the full sandbox timeout, undoing that fix."""
    from unittest.mock import MagicMock
    import time as _time

    monkeypatch.chdir(tmp_path)

    gui_code = (
        "import tkinter as tk\n"
        "root = tk.Tk()\n"
        "root.mainloop()\n"
    )
    agent = MagicMock()
    agent.run_agentic_loop.return_value = {
        "success": True, "code": gui_code, "stdout": "", "stderr": "",
        "attempts": 1, "iterations": 1, "history": [],
    }

    t0 = _time.time()
    result = run_code_generation_task(agent, "make a gui app", gen_tokens=64)
    elapsed = _time.time() - t0

    assert elapsed < 2, f"took {elapsed}s — looks like the code got re-executed for real"
    agent.run_in_sandbox.assert_not_called()
    assert result["executed"] is True
    assert result["success"] is True


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



