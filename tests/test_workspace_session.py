"""
Unit tests for chakra WorkspaceIndexer and SessionManager.
"""

import json
from pathlib import Path
import tempfile
import unittest

from chakra import SessionManager, WorkspaceIndexer


class TestWorkspaceIndexer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        # Create dummy python file
        py_file = self.root / "sample.py"
        py_file.write_text(
            "def foo(a: int, b: str = 'hello') -> bool:\n"
            "    return True\n\n"
            "class MyClass:\n"
            "    def bar(self, x: float) -> None:\n"
            "        pass\n",
            encoding="utf-8",
        )

        # Create dummy json file
        json_file = self.root / "config.json"
        json_file.write_text(json.dumps({"app_name": "test", "version": "1.0"}), encoding="utf-8")

        # Create dummy markdown file
        md_file = self.root / "README.md"
        md_file.write_text("# Project Title\n## Overview\nSome text.", encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_scan_workspace(self):
        indexer = WorkspaceIndexer(root_dir=self.root)
        index = indexer.scan_workspace()

        self.assertEqual(index["file_count"], 3)
        self.assertIn("sample.py", index["files"])
        self.assertIn("config.json", index["files"])
        self.assertIn("README.md", index["files"])

        py_data = index["files"]["sample.py"]["python_ast"]
        self.assertEqual(len(py_data["functions"]), 1)
        self.assertEqual(py_data["functions"][0]["name"], "foo")
        self.assertEqual(len(py_data["classes"]), 1)
        self.assertEqual(py_data["classes"][0]["name"], "MyClass")

    def test_get_tree(self):
        indexer = WorkspaceIndexer(root_dir=self.root)
        tree_str = indexer.get_tree()
        self.assertIn("sample.py", tree_str)
        self.assertIn("config.json", tree_str)
        self.assertIn("README.md", tree_str)

    def test_get_context_summary(self):
        indexer = WorkspaceIndexer(root_dir=self.root)
        summary = indexer.get_context_summary()
        self.assertIn("# Workspace Index", summary)
        self.assertIn("foo(a: int, b: str", summary)
        self.assertIn("class MyClass", summary)


class TestSessionManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage = Path(self.temp_dir.name) / "sessions"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_and_load_session(self):
        sm = SessionManager(storage_dir=self.storage)
        history = [{"role": "user", "content": "hello"}]
        artifacts = [{"file": "main.py", "code": "print('hello')"}]
        trace_logs = [{"agent": "CoderAgent", "action": "written"}]

        session_id = sm.save_session(
            session_id="test_sess_1",
            history=history,
            artifacts=artifacts,
            trace_logs=trace_logs,
            metadata={"user": "test_user"},
        )
        self.assertEqual(session_id, "test_sess_1")

        loaded = sm.load_session("test_sess_1")
        self.assertEqual(loaded["session_id"], "test_sess_1")
        self.assertEqual(loaded["history"], history)
        self.assertEqual(loaded["artifacts"], artifacts)
        self.assertEqual(loaded["trace_logs"], trace_logs)

    def test_list_sessions(self):
        sm = SessionManager(storage_dir=self.storage)
        sm.save_session(session_id="sess_a", history=[{"role": "user", "content": "a"}])
        sm.save_session(session_id="sess_b", history=[{"role": "user", "content": "b"}])

        sessions = sm.list_sessions()
        self.assertEqual(len(sessions), 2)
        session_ids = [s["session_id"] for s in sessions]
        self.assertIn("sess_a", session_ids)
        self.assertIn("sess_b", session_ids)

    def test_delete_session(self):
        sm = SessionManager(storage_dir=self.storage)
        sm.save_session(session_id="sess_del")
        self.assertTrue(sm.delete_session("sess_del"))
        self.assertFalse(sm.delete_session("sess_del"))
        with self.assertRaises(FileNotFoundError):
            sm.load_session("sess_del")


if __name__ == "__main__":
    unittest.main()
