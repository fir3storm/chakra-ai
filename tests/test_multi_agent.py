"""
Unit tests for Multi-Agent Orchestration system in chakra.multi_agent.
"""

import ast
import tempfile
import unittest
from pathlib import Path

from chakra.multi_agent import (
    ArchitectAgent,
    AuditorAgent,
    CoderAgent,
    MultiAgentOrchestrator,
)


class TestArchitectAgent(unittest.TestCase):
    def setUp(self):
        self.architect = ArchitectAgent()

    def test_plan_blueprint_calculator(self):
        blueprint = self.architect.plan_blueprint("Build a Python calculator application")
        self.assertIn("project_name", blueprint)
        self.assertIn("architecture_summary", blueprint)
        self.assertIn("modules", blueprint)
        self.assertGreaterEqual(len(blueprint["modules"]), 2)
        filenames = [m["filename"] for m in blueprint["modules"]]
        self.assertIn("config.py", filenames)
        self.assertIn("main.py", filenames)

    def test_plan_blueprint_port_scanner(self):
        blueprint = self.architect.plan_blueprint("Create a network port scanner security tool")
        self.assertIn("modules", blueprint)
        filenames = [m["filename"] for m in blueprint["modules"]]
        self.assertIn("scanner.py", filenames)

    def test_plan_blueprint_empty(self):
        blueprint = self.architect.plan_blueprint("")
        self.assertIn("modules", blueprint)
        self.assertGreater(len(blueprint["modules"]), 0)


class TestCoderAgent(unittest.TestCase):
    def setUp(self):
        self.coder = CoderAgent()

    def test_generate_module_config(self):
        spec = {
            "module_name": "config",
            "filename": "config.py",
            "purpose": "Application configuration settings",
            "specifications": "Define APP_NAME and load_config()",
            "dependencies": [],
            "interface": ["APP_NAME", "load_config"],
        }
        code = self.coder.generate_module(spec)
        self.assertIn("config.py", code)
        # Check syntax validity
        tree = ast.parse(code)
        self.assertIsNotNone(tree)

    def test_generate_module_main(self):
        spec = {
            "module_name": "main",
            "filename": "main.py",
            "purpose": "Application entry point",
            "specifications": "Execute main function",
            "dependencies": ["config.py"],
            "interface": ["main"],
        }
        context = {"config.py": "APP_NAME = 'TestApp'\n"}
        code = self.coder.generate_module(spec, context_modules=context)
        self.assertIn("main", code)
        tree = ast.parse(code)
        self.assertIsNotNone(tree)


class TestAuditorAgent(unittest.TestCase):
    def setUp(self):
        self.auditor = AuditorAgent()

    def test_audit_security_clean_code(self):
        code = (
            "def add(a, b):\n"
            "    return a + b\n\n"
            "print('Result:', add(5, 10))\n"
        )
        report = self.auditor.audit_security(code)
        self.assertEqual(report["score"], 100)
        self.assertFalse(report["has_critical"])
        self.assertEqual(len(report["issues"]), 0)

    def test_audit_security_eval_warning(self):
        code = "result = eval('2 + 2')\n"
        report = self.auditor.audit_security(code)
        self.assertTrue(report["has_critical"])
        self.assertLess(report["score"], 100)
        self.assertTrue(any("eval" in issue["message"] for issue in report["issues"]))

    def test_audit_and_test_execution(self):
        code = "import sys\nsys.stdout.write('AUDITOR_TEST_OK')\n"
        report = self.auditor.audit_and_test(code)
        self.assertTrue(report["passed"])
        self.assertEqual(report["exit_code"], 0)
        self.assertEqual(report["stdout"], "AUDITOR_TEST_OK")


class TestMultiAgentOrchestrator(unittest.TestCase):
    def setUp(self):
        self.orchestrator = MultiAgentOrchestrator()

    def test_run_team_task_calculator(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = self.orchestrator.run_team_task(
                prompt="Build a modular calculator",
                output_dir=tmp_dir,
                max_retries=2,
            )
            self.assertTrue(res["success"])
            self.assertIn("blueprint", res)
            self.assertIn("modules", res)
            self.assertIn("audit_reports", res)
            self.assertEqual(res["output_dir"], str(Path(tmp_dir).resolve()))

            # Verify files exist in directory
            for filename in res["modules"].keys():
                filepath = Path(tmp_dir) / filename
                self.assertTrue(filepath.exists())
                self.assertGreater(filepath.stat().st_size, 0)

    def test_run_team_task_port_scanner(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = self.orchestrator.run_team_task(
                prompt="Build a network port scanner",
                output_dir=tmp_dir,
                max_retries=2,
            )
            self.assertTrue(res["success"])
            self.assertIn("scanner.py", res["modules"])
            self.assertTrue((Path(tmp_dir) / "scanner.py").exists())


class ScriptedGrammarBackend:
    """Fake engine-C-shaped backend (has generate_choice/generate_json/generate) driven by a
    scripted sequence, so the tool-driven agent paths can be tested without a real GGUF model."""

    def __init__(self, decisions, calls, final_answer):
        self.decisions = list(decisions)
        self.calls = list(calls)
        self.final_answer = final_answer
        self._decision_i = 0
        self._call_i = 0

    def generate_choice(self, choices=None, prompt=None, messages=None, system=""):
        i = min(self._decision_i, len(self.decisions) - 1)
        self._decision_i += 1
        return self.decisions[i]

    def generate_json(self, schema=None, prompt=None, messages=None, system="", n_new=256):
        i = min(self._call_i, len(self.calls) - 1)
        self._call_i += 1
        return self.calls[i]

    def generate(self, prompt=None, messages=None, **kw):
        return self.final_answer


class TestToolDrivenAgents(unittest.TestCase):
    """Covers the new grammar-constrained tool-calling path added to ArchitectAgent/CoderAgent.
    Uses ScriptedGrammarBackend so these run fast/deterministically without a real model."""

    def test_architect_uses_tools_then_parses_blueprint(self):
        blueprint_json = (
            '{"project_name": "demo", "architecture_summary": "s", '
            '"modules": [{"module_name": "main", "filename": "main.py", "purpose": "p", '
            '"specifications": "s", "dependencies": [], "interface": ["main"]}]}'
        )
        backend = ScriptedGrammarBackend(
            decisions=["CALL_TOOL", "FINAL_ANSWER"],
            calls=[{"tool": "list_dir", "args": {}}],
            final_answer=blueprint_json,
        )
        architect = ArchitectAgent(model=backend)
        blueprint = architect.plan_blueprint("build a small tool", workspace_root=".")
        self.assertEqual(blueprint["project_name"], "demo")
        self.assertEqual(len(architect.tool_call_log), 1)
        self.assertEqual(architect.tool_call_log[0]["tool"], "list_dir")

    def test_architect_falls_back_to_template_when_tool_output_unparseable(self):
        backend = ScriptedGrammarBackend(
            decisions=["FINAL_ANSWER"],
            calls=[],
            final_answer="not json at all",
        )
        architect = ArchitectAgent(model=backend)
        blueprint = architect.plan_blueprint("build a calculator")
        # No valid JSON from the model at all -> last-resort template fallback still fires.
        self.assertIn("modules", blueprint)
        filenames = [m["filename"] for m in blueprint["modules"]]
        self.assertIn("main.py", filenames)

    def test_coder_forces_read_before_patch_on_retry(self, tmp_dir=None):
        with tempfile.TemporaryDirectory() as tmp_dir:
            (Path(tmp_dir) / "engine.py").write_text(
                "def add(a, b):\n    return a - b\n", encoding="utf-8"
            )
            backend = ScriptedGrammarBackend(
                decisions=["CALL_TOOL", "FINAL_ANSWER"],
                calls=[{
                    "tool": "edit_file",
                    "args": {"path": "engine.py", "search": "return a - b", "replace": "return a + b"},
                }],
                final_answer="```python\ndef add(a, b):\n    return a + b\n```",
            )
            coder = CoderAgent(model=backend)
            spec = {
                "module_name": "engine", "filename": "engine.py", "purpose": "adder",
                "specifications": "add(a,b)", "dependencies": [],
            }
            code = coder.generate_module(
                spec, context_modules={}, feedback="add(2,3) returned -1", workspace_root=tmp_dir
            )
            self.assertIn("return a + b", code)
            # forced_first_call (read_file) plus the scripted edit_file call.
            tools_called = [tc["tool"] for tc in coder.tool_call_log]
            self.assertEqual(tools_called, ["read_file", "edit_file"])
            self.assertTrue((Path(tmp_dir) / "engine.py").read_text(encoding="utf-8"))

    def test_coder_falls_back_to_template_when_no_model(self):
        coder = CoderAgent(model=None)
        spec = {"module_name": "config", "filename": "config.py", "purpose": "settings",
                "specifications": "", "dependencies": []}
        code = coder.generate_module(spec, context_modules={}, feedback=None, workspace_root=None)
        ast.parse(code)  # last-resort template is always syntactically valid
        self.assertIn("APP_NAME", code)


if __name__ == "__main__":
    unittest.main()
