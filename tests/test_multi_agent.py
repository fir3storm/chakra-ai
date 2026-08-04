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


if __name__ == "__main__":
    unittest.main()
