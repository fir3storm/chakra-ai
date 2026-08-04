"""
Unit tests for advanced features in chakra:
WorkspaceIndexer, SessionManager, InfoSecAuditor, PersonaManager, and diff/runner functions.
"""

import os
from pathlib import Path
import tempfile
import unittest

from chakra import (
    InfoSecAuditor,
    PersonaManager,
    SessionManager,
    WorkspaceIndexer,
    apply_diff,
    execute_sandbox,
    generate_diff,
    run_in_sandbox,
)


class TestWorkspaceIndexer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)

        # Create dummy project structure
        (self.root_path / "src").mkdir()
        (self.root_path / "src" / "main.py").write_text(
            '"""Main entry."""\n\ndef add_numbers(a: int, b: int) -> int:\n    """Add two numbers."""\n    return a + b\n\nclass Calculator:\n    """Calculator class."""\n    def compute(self, x):\n        return x * 2\n',
            encoding="utf-8",
        )
        (self.root_path / "README.md").write_text("# Test Project\n\n## Overview\nDocumentation here.", encoding="utf-8")
        (self.root_path / "config.json").write_text('{"name": "test_app", "version": "1.0.0"}', encoding="utf-8")
        (self.root_path / "schema.sql").write_text("CREATE TABLE users (id INT PRIMARY KEY);", encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_scan_workspace_file_count(self):
        indexer = WorkspaceIndexer(root_dir=self.root_path)
        index = indexer.scan_workspace()
        self.assertTrue(indexer.scanned)
        self.assertGreaterEqual(index["file_count"], 4)
        self.assertIn("src/main.py", index["files"])

    def test_python_signature_extraction(self):
        indexer = WorkspaceIndexer(root_dir=self.root_path)
        index = indexer.scan_workspace()
        main_info = index["files"]["src/main.py"]
        ast_info = main_info.get("python_ast", {})

        functions = ast_info.get("functions", [])
        classes = ast_info.get("classes", [])

        self.assertEqual(len(functions), 1)
        self.assertEqual(functions[0]["name"], "add_numbers")
        self.assertIn("add_numbers", functions[0]["signature"])

        self.assertEqual(len(classes), 1)
        self.assertEqual(classes[0]["name"], "Calculator")
        self.assertEqual(len(classes[0]["methods"]), 1)
        self.assertEqual(classes[0]["methods"][0]["name"], "compute")

    def test_get_tree(self):
        indexer = WorkspaceIndexer(root_dir=self.root_path)
        tree_str = indexer.get_tree(max_depth=3)
        self.assertIn(self.root_path.name, tree_str)
        self.assertIn("src/", tree_str)

    def test_get_context_summary(self):
        indexer = WorkspaceIndexer(root_dir=self.root_path)
        summary = indexer.get_context_summary(max_tokens_approx=1000)
        self.assertIn("# Workspace Index", summary)
        self.assertIn("Key Python Signatures", summary)
        self.assertIn("add_numbers", summary)

    def test_nonexistent_directory(self):
        non_existent = self.root_path / "non_existent_folder"
        indexer = WorkspaceIndexer(root_dir=non_existent)
        index = indexer.scan_workspace()
        self.assertEqual(index["file_count"], 0)
        self.assertEqual(len(index["files"]), 0)


class TestSessionManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.session_mgr = SessionManager(storage_dir=self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_and_load_session(self):
        history = [
            {"role": "user", "content": "Build a web app"},
            {"role": "assistant", "content": "Creating blueprint..."},
        ]
        artifacts = [{"filename": "app.py", "code": "print('hello')"}]
        trace_logs = [{"step": 1, "agent": "Architect", "status": "OK"}]
        metadata = {"project": "WebApp"}

        session_id = self.session_mgr.save_session(
            history=history,
            artifacts=artifacts,
            trace_logs=trace_logs,
            metadata=metadata,
        )
        self.assertIsNotNone(session_id)

        loaded = self.session_mgr.load_session(session_id)
        self.assertEqual(loaded["session_id"], session_id)
        self.assertEqual(len(loaded["history"]), 2)
        self.assertEqual(loaded["history"][0]["content"], "Build a web app")
        self.assertEqual(loaded["metadata"]["project"], "WebApp")

    def test_list_sessions(self):
        self.session_mgr.save_session(session_id="session_1", history=[{"role": "user", "content": "Hi"}])
        self.session_mgr.save_session(session_id="session_2", history=[{"role": "user", "content": "Hello"}])

        sessions = self.session_mgr.list_sessions()
        self.assertEqual(len(sessions), 2)
        session_ids = [s["session_id"] for s in sessions]
        self.assertIn("session_1", session_ids)
        self.assertIn("session_2", session_ids)

    def test_delete_session(self):
        s_id = self.session_mgr.save_session(session_id="session_to_delete")
        self.assertTrue(self.session_mgr.delete_session(s_id))
        self.assertFalse(self.session_mgr.delete_session(s_id))
        with self.assertRaises(FileNotFoundError):
            self.session_mgr.load_session(s_id)


class TestInfoSecAuditor(unittest.TestCase):
    def setUp(self):
        self.auditor = InfoSecAuditor()

    def test_audit_clean_code(self):
        clean_code = (
            "def safe_multiply(a: int, b: int) -> int:\n"
            "    return a * b\n\n"
            "print(safe_multiply(5, 10))\n"
        )
        report = self.auditor.audit_code(clean_code)
        self.assertEqual(report["score"], 100)
        self.assertTrue(report["grade"].startswith("A"))
        self.assertEqual(report["status"], "PASSED")
        self.assertEqual(len(report["vulnerabilities"]), 0)

    def test_audit_unsafe_eval(self):
        unsafe_code = "user_input = input()\nresult = eval(user_input)\n"
        report = self.auditor.audit_code(unsafe_code)
        self.assertLess(report["score"], 100)
        self.assertTrue(any(v["rule_id"] in ("SEC-CODE-01", "SEC-EVAL-EXEC") for v in report["vulnerabilities"]))

    def test_audit_file_and_workspace(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            f1 = tmp_path / "safe.py"
            f1.write_text("print('Safe module')\n", encoding="utf-8")
            f2 = tmp_path / "unsafe.py"
            f2.write_text("import os\nos.system('dir')\n", encoding="utf-8")

            file_report = self.auditor.audit_file(f1)
            self.assertEqual(file_report["score"], 100)

            ws_report = self.auditor.scan_workspace(tmp_path)
            self.assertEqual(ws_report["scanned_files"], 2)
            self.assertIn("file_reports", ws_report)


class TestPersonaManager(unittest.TestCase):
    def setUp(self):
        self.manager = PersonaManager(initial_persona="infosec")

    def test_default_personas(self):
        personas = self.manager.list_personas()
        self.assertIn("infosec", personas)
        self.assertIn("architect", personas)
        self.assertIn("devops", personas)
        self.assertIn("fullstack", personas)

    def test_switch_persona(self):
        active = self.manager.get_active_persona()
        self.assertEqual(active["role"], "infosec")

        self.manager.set_persona("architect")
        new_active = self.manager.get_active_persona()
        self.assertEqual(new_active["role"], "architect")

    def test_invalid_persona_raises(self):
        with self.assertRaises(ValueError):
            self.manager.set_persona("non_existent_role")

    def test_get_system_prompt(self):
        prompt = self.manager.get_system_prompt(custom_instructions="Focus on API security.")
        self.assertIn("InfoSec Security Audit", prompt)
        self.assertIn("Focus on API security.", prompt)

    def test_add_custom_persona(self):
        self.manager.add_custom_persona(
            role="tester",
            name="QA Testing Specialist",
            description="Specialist in unit testing and TDD.",
            system_prompt="You are QA testing specialist.",
        )
        self.manager.set_persona("tester")
        active = self.manager.get_active_persona()
        self.assertEqual(active["role"], "tester")
        self.assertEqual(active["name"], "QA Testing Specialist")


class TestDiffAndRunnerFunctions(unittest.TestCase):
    def test_generate_diff(self):
        old_code = "def foo():\n    return 1\n"
        new_code = "def foo():\n    return 2\n"
        diff_str = generate_diff(old_code, new_code)
        self.assertIn("-    return 1", diff_str)
        self.assertIn("+    return 2", diff_str)

    def test_apply_diff(self):
        old_code = "def foo():\n    return 1\n"
        new_code = "def foo():\n    return 2\n"
        diff_str = generate_diff(old_code, new_code)
        patched = apply_diff(old_code, diff_str)
        self.assertIn("return 2", patched)

    def test_run_in_sandbox(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("import sys\nsys.stdout.write('SANDBOX_RUNNER_OK')\n")
            tmp_path = f.name

        try:
            res = run_in_sandbox(tmp_path, timeout=5)
            self.assertTrue(res["success"])
            self.assertEqual(res["exit_code"], 0)
            self.assertEqual(res["stdout"], "SANDBOX_RUNNER_OK")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_execute_sandbox(self):
        code = "import sys\nsys.stdout.write('EXECUTE_SANDBOX_OK')\n"
        res = execute_sandbox(code, timeout=5)
        self.assertTrue(res["success"])
        self.assertEqual(res["returncode"], 0)
        self.assertEqual(res["stdout"], "EXECUTE_SANDBOX_OK")


if __name__ == "__main__":
    unittest.main()
