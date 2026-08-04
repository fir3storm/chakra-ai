"""
Unit tests for InfoSecAuditor (kimipy.security) and PersonaManager (kimipy.persona).
"""

import tempfile
from pathlib import Path
import unittest

from kimipy import InfoSecAuditor, PersonaManager


class TestInfoSecAuditor(unittest.TestCase):
    """Test suite for InfoSecAuditor security audit engine."""

    def setUp(self) -> None:
        self.auditor = InfoSecAuditor()

    def test_clean_code_audit(self) -> None:
        clean_code = """
def add(a: int, b: int) -> int:
    return a + b
"""
        res = self.auditor.audit_code(clean_code)
        self.assertEqual(res["score"], 100)
        self.assertEqual(res["grade"], "A+")
        self.assertEqual(res["status"], "PASSED")
        self.assertEqual(res["summary"]["total"], 0)

    def test_hardcoded_credentials(self) -> None:
        vulnerable_code = """
AWS_SECRET_KEY = "AKIAIOSFODNN7EXAMPLE"
db_password = "change_me_before_production"
"""
        res = self.auditor.audit_code(vulnerable_code)
        self.assertLess(res["score"], 100)
        self.assertGreater(res["summary"]["total"], 0)
        vuln_types = [v["type"] for v in res["vulnerabilities"]]
        self.assertIn("Hardcoded Credentials", vuln_types)

    def test_unsafe_eval_exec(self) -> None:
        vulnerable_code = """
user_input = "__import__('os').system('dir')"
eval(user_input)
exec("print(1)")
"""
        res = self.auditor.audit_code(vulnerable_code)
        self.assertEqual(res["status"], "FAILED")
        vuln_rules = [v["rule_id"] for v in res["vulnerabilities"]]
        self.assertIn("SEC-CODE-01", vuln_rules)

    def test_command_injection(self) -> None:
        vulnerable_code = """
import os
import subprocess

cmd = "dir " + input_user
os.system(cmd)
subprocess.run(cmd, shell=True)
"""
        res = self.auditor.audit_code(vulnerable_code)
        vuln_rules = [v["rule_id"] for v in res["vulnerabilities"]]
        self.assertTrue(any(r in ("SEC-CMD-01", "SEC-CMD-02") for r in vuln_rules))

    def test_sql_injection(self) -> None:
        vulnerable_code = """
def search_user(cursor, username):
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)
"""
        res = self.auditor.audit_code(vulnerable_code)
        vuln_rules = [v["rule_id"] for v in res["vulnerabilities"]]
        self.assertIn("SEC-SQL-01", vuln_rules)

    def test_weak_crypto(self) -> None:
        vulnerable_code = """
import hashlib
import random

def get_hash(data):
    return hashlib.md5(data.encode()).hexdigest()

def generate_token():
    return str(random.randint(1000, 9999))  # random used for token
"""
        res = self.auditor.audit_code(vulnerable_code)
        vuln_types = [v["type"] for v in res["vulnerabilities"]]
        self.assertIn("Weak Hashing Algorithm", vuln_types)
        self.assertIn("Insecure Randomness", vuln_types)

    def test_scan_workspace_and_format_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            file1 = tmp_path / "clean.py"
            file1.write_text("def ok(): return True\n", encoding="utf-8")

            file2 = tmp_path / "unsafe.py"
            file2.write_text("eval('1+1')\n", encoding="utf-8")

            scan_res = self.auditor.scan_workspace(tmp_path)
            self.assertEqual(scan_res["scanned_files"], 2)
            self.assertIn("formatted_report", scan_res)
            self.assertIn("INFOSEC SECURITY AUDIT REPORT", scan_res["formatted_report"])


class TestPersonaManager(unittest.TestCase):
    """Test suite for PersonaManager."""

    def test_default_persona_initialization(self) -> None:
        pm = PersonaManager()
        active = pm.get_active_persona()
        self.assertEqual(active["role"], "infosec")
        self.assertEqual(active["name"], "InfoSec Expert & Security Auditor")

    def test_persona_switching(self) -> None:
        pm = PersonaManager()
        roles = ["architect", "devops", "fullstack", "infosec"]
        for role in roles:
            set_role = pm.set_persona(role)
            self.assertEqual(set_role, role)
            self.assertEqual(pm.get_active_persona()["role"], role)

    def test_invalid_persona_raise(self) -> None:
        pm = PersonaManager()
        with self.assertRaises(ValueError):
            pm.set_persona("non_existent_role")

    def test_list_personas(self) -> None:
        pm = PersonaManager()
        all_personas = pm.list_personas()
        self.assertIn("infosec", all_personas)
        self.assertIn("architect", all_personas)
        self.assertIn("devops", all_personas)
        self.assertIn("fullstack", all_personas)

    def test_get_system_prompt_adaptation(self) -> None:
        pm = PersonaManager("architect")
        prompt = pm.get_system_prompt(custom_instructions="Focus on Microservices")
        self.assertIn("System Architect Expert", prompt)
        self.assertIn("Focus on Microservices", prompt)

    def test_add_custom_persona(self) -> None:
        pm = PersonaManager()
        pm.add_custom_persona(
            role="quantum",
            name="Quantum Computing Specialist",
            description="Specialist in Qiskit and quantum algorithms.",
            system_prompt="You are a Quantum Computing Expert."
        )
        pm.set_persona("quantum")
        active = pm.get_active_persona()
        self.assertEqual(active["role"], "quantum")
        self.assertEqual(active["name"], "Quantum Computing Specialist")


if __name__ == "__main__":
    unittest.main()
