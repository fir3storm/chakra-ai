# chakra/security.py
"""
InfoSec Security Audit Engine for Kimi K3 PyTorch Runtime on Windows.
Author & Creator: Abhirup Guha (Info Security Solution)

Provides InfoSecAuditor for static AST security analysis checking OWASP vulnerabilities
(hardcoded credentials, SQL injection, unsafe eval/exec, command injection, weak crypto),
computing 0-100 security scores, and formatting comprehensive audit reports.
"""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union


class _SecurityASTVisitor(ast.NodeVisitor):
    """
    AST Visitor to detect OWASP security vulnerabilities in Python code.
    """

    # Regex for potential credential variable/key names
    CRED_PATTERN = re.compile(
        r"(?i)^(.*password.*|.*passwd.*|.*secret.*|.*api_?key.*|.*token.*|.*auth_?token.*|.*jwt.*|.*private_?key.*|.*access_?key.*|.*db_pass.*)$"
    )

    # Placeholders to ignore when checking hardcoded strings
    PLACEHOLDERS: Set[str] = {
        "",
        "todo",
        "your_key_here",
        "your_password_here",
        "xxxx",
        "xxxxxx",
        "change_me",
        "placeholder",
        "none",
        "null",
        "env",
        "os.environ",
        "config",
        "test",
        "example",
    }

    # SQL keywords pattern
    SQL_PATTERN = re.compile(r"(?i)\b(SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|EXEC|UNION)\b")

    def __init__(self, source_lines: List[str]) -> None:
        self.source_lines = source_lines
        self.vulnerabilities: List[Dict[str, Any]] = []
        self._vuln_counter = 0

    def _add_vuln(
        self,
        rule_id: str,
        vuln_type: str,
        severity: str,
        line: int,
        description: str,
        recommendation: str,
    ) -> None:
        self._vuln_counter += 1
        snippet = ""
        if 1 <= line <= len(self.source_lines):
            snippet = self.source_lines[line - 1].strip()

        self.vulnerabilities.append({
            "id": f"SEC-{self._vuln_counter:03d}",
            "rule_id": rule_id,
            "type": vuln_type,
            "severity": severity.upper(),
            "line": line,
            "description": description,
            "snippet": snippet,
            "recommendation": recommendation,
        })

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            var_name = self._get_target_name(target)
            if var_name and self.CRED_PATTERN.match(var_name):
                val = self._extract_constant_str(node.value)
                if val is not None and val.lower() not in self.PLACEHOLDERS and len(val) > 1:
                    sev = "HIGH" if any(k in var_name.lower() for k in ["secret", "private", "jwt", "token"]) else "MEDIUM"
                    self._add_vuln(
                        rule_id="SEC-CRED-01",
                        vuln_type="Hardcoded Credentials",
                        severity=sev,
                        line=node.lineno,
                        description=f"Hardcoded sensitive credential detected in variable assignment '{var_name}'.",
                        recommendation="Store credentials in environment variables or a secure secret vault.",
                    )
            # Check for SQL injection in dynamic query variable assignments
            if var_name and any(k in var_name.lower() for k in ("sql", "query", "stmt")):
                if self._is_dynamic_string(node.value):
                    self._add_vuln(
                        rule_id="SEC-SQL-01",
                        vuln_type="SQL Injection",
                        severity="HIGH",
                        line=node.lineno,
                        description=f"Dynamic SQL query construction detected in variable assignment '{var_name}'.",
                        recommendation="Use parameterized SQL queries with placeholder parameters instead of string formatting.",
                    )
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        for key_node, val_node in zip(node.keys, node.values):
            if key_node is not None:
                key_str = self._extract_constant_str(key_node)
                if key_str and self.CRED_PATTERN.match(key_str):
                    val_str = self._extract_constant_str(val_node)
                    if val_str is not None and val_str.lower() not in self.PLACEHOLDERS and len(val_str) > 1:
                        sev = "HIGH" if any(k in key_str.lower() for k in ["secret", "private", "jwt", "token"]) else "MEDIUM"
                        self._add_vuln(
                            rule_id="SEC-CRED-02",
                            vuln_type="Hardcoded Credentials in Dict",
                            severity=sev,
                            line=node.lineno,
                            description=f"Hardcoded secret credential detected in dictionary key '{key_str}'.",
                            recommendation="Load dictionary secrets dynamically from external environment configuration.",
                        )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func_name = self._get_func_name(node.func)

        # 1. Unsafe eval / exec
        if func_name in ("eval", "exec", "builtins.eval", "builtins.exec"):
            self._add_vuln(
                rule_id="SEC-CODE-01",
                vuln_type="Unsafe Code Execution",
                severity="HIGH",
                line=node.lineno,
                description=f"Use of unsafe function '{func_name}' allows arbitrary Python code execution.",
                recommendation="Avoid eval/exec. Use ast.literal_eval or structured data formats (e.g. JSON).",
            )

        # 2. Command injection via os.system, os.popen, subprocess
        if func_name in ("os.system", "os.popen"):
            if node.args:
                arg0 = node.args[0]
                if not isinstance(arg0, ast.Constant):
                    self._add_vuln(
                        rule_id="SEC-CMD-01",
                        vuln_type="Command Injection",
                        severity="HIGH",
                        line=node.lineno,
                        description=f"Unsafe invocation of '{func_name}' with dynamic command argument.",
                        recommendation="Use subprocess.run with shell=False and pass arguments as an explicit sequence.",
                    )
        elif func_name in (
            "subprocess.Popen",
            "subprocess.run",
            "subprocess.call",
            "subprocess.check_output",
            "subprocess.check_call",
            "asyncio.create_subprocess_shell",
        ):
            shell_true = False
            for kw in node.keywords:
                if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and bool(kw.value.value):
                    shell_true = True
                    break
            if shell_true or func_name == "asyncio.create_subprocess_shell":
                self._add_vuln(
                    rule_id="SEC-CMD-02",
                    vuln_type="Command Injection (shell=True)",
                    severity="HIGH",
                    line=node.lineno,
                    description=f"Execution of shell command via '{func_name}' with shell=True enabled.",
                    recommendation="Set shell=False and pass command line arguments as a list of strings.",
                )

        # 3. SQL Injection detection
        if any(kw in func_name.lower() for kw in ("execute", "executemany", "raw", "query")):
            if node.args:
                query_arg = node.args[0]
                is_dyn = self._is_dynamic_string(query_arg)
                is_sql_var = isinstance(query_arg, ast.Name) and any(k in query_arg.id.lower() for k in ("sql", "query", "stmt"))
                if is_dyn or is_sql_var:
                    if not any(v["line"] == node.lineno and v["rule_id"] == "SEC-SQL-01" for v in self.vulnerabilities):
                        self._add_vuln(
                            rule_id="SEC-SQL-01",
                            vuln_type="SQL Injection",
                            severity="HIGH",
                            line=node.lineno,
                            description=f"Dynamic SQL query passed to database query method call '{func_name}'.",
                            recommendation="Use parameterized SQL queries with query parameter bindings.",
                        )

        # 4. Weak Cryptography
        if func_name in ("hashlib.md5", "hashlib.sha1"):
            self._add_vuln(
                rule_id="SEC-CRYPTO-01",
                vuln_type="Weak Hashing Algorithm",
                severity="MEDIUM",
                line=node.lineno,
                description=f"Use of cryptographically broken hashing algorithm '{func_name}'.",
                recommendation="Use secure cryptographic hashes such as hashlib.sha256 or hashlib.sha512.",
            )
        elif func_name in ("Crypto.Cipher.DES.new", "Crypto.Cipher.ARC4.new", "Crypto.Cipher.Blowfish.new"):
            self._add_vuln(
                rule_id="SEC-CRYPTO-02",
                vuln_type="Weak Cipher Algorithm",
                severity="HIGH",
                line=node.lineno,
                description=f"Use of weak encryption cipher algorithm in '{func_name}'.",
                recommendation="Use AES-GCM or AES-256 (AES.new) with authenticated encryption.",
            )
        elif func_name.startswith("random."):
            line_str = self.source_lines[node.lineno - 1].lower() if 1 <= node.lineno <= len(self.source_lines) else ""
            if any(sec_kw in line_str for sec_kw in ["token", "password", "key", "secret", "session", "salt"]):
                self._add_vuln(
                    rule_id="SEC-CRYPTO-03",
                    vuln_type="Insecure Randomness",
                    severity="LOW",
                    line=node.lineno,
                    description=f"Pseudo-random generator '{func_name}' used in security-sensitive context.",
                    recommendation="Use the cryptographically secure 'secrets' module (e.g., secrets.token_hex).",
                )

        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name in ("crypt", "Crypto.Cipher.DES", "Crypto.Cipher.ARC4"):
                self._add_vuln(
                    rule_id="SEC-CRYPTO-04",
                    vuln_type="Weak Crypto Import",
                    severity="MEDIUM",
                    line=node.lineno,
                    description=f"Import of legacy or weak cryptographic library '{alias.name}'.",
                    recommendation="Migrate to modern cryptography primitives (e.g. cryptography package).",
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module in ("crypt", "Crypto.Cipher.DES", "Crypto.Cipher.ARC4"):
            self._add_vuln(
                rule_id="SEC-CRYPTO-04",
                vuln_type="Weak Crypto Import",
                severity="MEDIUM",
                line=node.lineno,
                description=f"Import from legacy cryptographic module '{node.module}'.",
                recommendation="Migrate to modern cryptography primitives (e.g. cryptography package).",
            )
        self.generic_visit(node)

    def _get_target_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return ""

    def _extract_constant_str(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    def _get_func_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            val = self._get_func_name(node.value)
            return f"{val}.{node.attr}" if val else node.attr
        return ""

    def _is_dynamic_string(self, node: ast.AST) -> bool:
        if isinstance(node, ast.JoinedStr):
            return True
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, (ast.Mod, ast.Add)):
                return True
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "format":
                return True
        return False


class InfoSecAuditor:
    """
    InfoSecAuditor: Specialized security audit engine for Info Security Solution.
    Performs static AST analysis checking for OWASP vulnerabilities (hardcoded credentials,
    SQL injection, unsafe eval/exec, command injection, weak crypto), computes 0-100 security score,
    and formats comprehensive vulnerability reports.
    """

    PENALTIES = {
        "HIGH": 20,
        "MEDIUM": 10,
        "LOW": 5,
    }

    def __init__(self, custom_rules: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize InfoSecAuditor engine.

        Args:
            custom_rules: Optional dictionary of custom audit configuration rules.
        """
        self.custom_rules = custom_rules or {}

    def audit_code(self, code: str, filename: str = "<string>") -> Dict[str, Any]:
        """
        Audits Python source code string for OWASP security vulnerabilities.

        Args:
            code: Python code string.
            filename: Label for the code source file.

        Returns:
            Dict containing file audit results, vulnerabilities, security score, grade, and status.
        """
        source_lines = code.splitlines()
        vulnerabilities: List[Dict[str, Any]] = []
        syntax_error: Optional[str] = None

        try:
            tree = ast.parse(code, filename=filename)
            visitor = _SecurityASTVisitor(source_lines)
            visitor.visit(tree)
            vulnerabilities = visitor.vulnerabilities
        except SyntaxError as se:
            syntax_error = f"Syntax error at line {se.lineno}: {se.msg}"
            vulnerabilities.append({
                "id": "SEC-ERR-001",
                "rule_id": "SEC-SYNTAX-ERROR",
                "type": "Syntax Error",
                "severity": "HIGH",
                "line": se.lineno or 1,
                "description": f"Failed to parse AST: {se.msg}",
                "snippet": se.text.strip() if se.text else "",
                "recommendation": "Fix syntax errors before running security audit.",
            })

        # Calculate security score (100 base)
        penalty = sum(self.PENALTIES.get(v["severity"], 10) for v in vulnerabilities)
        score = max(0, min(100, 100 - penalty))

        grade, status = self._compute_grade_and_status(score, vulnerabilities, syntax_error)

        high_cnt = sum(1 for v in vulnerabilities if v["severity"] == "HIGH")
        med_cnt = sum(1 for v in vulnerabilities if v["severity"] == "MEDIUM")
        low_cnt = sum(1 for v in vulnerabilities if v["severity"] == "LOW")

        return {
            "file_path": filename,
            "score": score,
            "grade": grade,
            "status": status,
            "vulnerabilities": vulnerabilities,
            "summary": {
                "total": len(vulnerabilities),
                "high": high_cnt,
                "medium": med_cnt,
                "low": low_cnt,
            },
            "syntax_error": syntax_error,
        }

    def audit_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Audits a single Python file for security vulnerabilities.

        Args:
            file_path: Path to the .py file.

        Returns:
            Dict containing audit report for the file.
        """
        path = Path(file_path)
        if not path.is_file():
            return {
                "file_path": str(file_path),
                "score": 0,
                "grade": "F",
                "status": "ERROR",
                "vulnerabilities": [],
                "summary": {"total": 0, "high": 0, "medium": 0, "low": 0},
                "syntax_error": f"File not found: {file_path}",
            }

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            return self.audit_code(content, filename=str(path))
        except Exception as e:
            return {
                "file_path": str(path),
                "score": 0,
                "grade": "F",
                "status": "ERROR",
                "vulnerabilities": [],
                "summary": {"total": 0, "high": 0, "medium": 0, "low": 0},
                "syntax_error": f"Error reading file: {e}",
            }

    def scan_workspace(
        self,
        workspace_dir: Union[str, Path],
        exclude_dirs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Recursively scans a workspace directory for security vulnerabilities across Python files.

        Args:
            workspace_dir: Root directory of the workspace.
            exclude_dirs: List of directory names to skip (e.g. ['.git', '__pycache__', '.venv']).

        Returns:
            Dict containing aggregated workspace security scan results and report.
        """
        root = Path(workspace_dir)
        if not root.is_dir():
            raise ValueError(f"Workspace directory does not exist: {workspace_dir}")

        if exclude_dirs is None:
            exclude_dirs = [".git", "__pycache__", ".venv", "venv", "node_modules", "build", "dist", ".pytest_cache", ".gemini"]

        exclude_set = set(exclude_dirs)
        file_reports: List[Dict[str, Any]] = []

        for current_root, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in exclude_set]
            for file in files:
                if file.endswith(".py"):
                    full_path = Path(current_root) / file
                    report = self.audit_file(full_path)
                    file_reports.append(report)

        if not file_reports:
            overall_score = 100
            severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        else:
            overall_score = int(sum(r["score"] for r in file_reports) / len(file_reports))
            severity_counts = {
                "HIGH": sum(r["summary"]["high"] for r in file_reports),
                "MEDIUM": sum(r["summary"]["medium"] for r in file_reports),
                "LOW": sum(r["summary"]["low"] for r in file_reports),
            }

        total_vulns = sum(severity_counts.values())
        overall_grade, overall_status = self._compute_grade_and_status(
            overall_score,
            [{"severity": "HIGH"} for _ in range(severity_counts["HIGH"])],
        )

        scan_result = {
            "workspace_dir": str(root),
            "scanned_files": len(file_reports),
            "total_vulnerabilities": total_vulns,
            "overall_score": overall_score,
            "overall_grade": overall_grade,
            "overall_status": overall_status,
            "severity_counts": severity_counts,
            "file_reports": file_reports,
        }

        scan_result["formatted_report"] = self.format_report(scan_result)
        return scan_result

    def format_report(self, scan_result: Dict[str, Any]) -> str:
        """
        Formats audit scan results into a human-readable CLI report string.

        Args:
            scan_result: Dictionary returned from audit_file or scan_workspace.

        Returns:
            Formatted report string.
        """
        lines: List[str] = []
        lines.append("=" * 72)
        lines.append("        INFOSEC SECURITY AUDIT REPORT - INFO SECURITY SOLUTION")
        lines.append("=" * 72)

        if "workspace_dir" in scan_result:
            lines.append(f"Workspace Path  : {scan_result['workspace_dir']}")
            lines.append(f"Scanned Files   : {scan_result['scanned_files']}")
            lines.append(f"Overall Score   : {scan_result['overall_score']}/100 (Grade: {scan_result['overall_grade']})")
            lines.append(f"Scan Status     : {scan_result['overall_status']}")
            sev = scan_result['severity_counts']
            lines.append(f"Vulnerabilities : Total={scan_result['total_vulnerabilities']} | HIGH={sev['HIGH']} | MEDIUM={sev['MEDIUM']} | LOW={sev['LOW']}")
            lines.append("-" * 72)

            for idx, r in enumerate(scan_result["file_reports"], 1):
                if r["vulnerabilities"]:
                    rel_path = r["file_path"]
                    lines.append(f"[{idx}] {rel_path} - Score: {r['score']}/100 ({r['grade']})")
                    for v in r["vulnerabilities"]:
                        lines.append(f"    - [{v['severity']}] Line {v['line']}: {v['type']} - {v['description']}")
                        if v.get("snippet"):
                            lines.append(f"      Snippet: {v['snippet']}")
                        lines.append(f"      Fix    : {v['recommendation']}")
                    lines.append("")
        else:
            lines.append(f"File Path       : {scan_result['file_path']}")
            lines.append(f"Security Score  : {scan_result['score']}/100 (Grade: {scan_result['grade']})")
            lines.append(f"Audit Status    : {scan_result['status']}")
            lines.append(f"Vulnerabilities : Total={scan_result['summary']['total']} | HIGH={scan_result['summary']['high']} | MEDIUM={scan_result['summary']['medium']} | LOW={scan_result['summary']['low']}")
            lines.append("-" * 72)

            for v in scan_result["vulnerabilities"]:
                lines.append(f"  * [{v['severity']}] Line {v['line']}: {v['type']} ({v['rule_id']})")
                lines.append(f"    Description: {v['description']}")
                if v.get("snippet"):
                    lines.append(f"    Code       : {v['snippet']}")
                lines.append(f"    Recommendation: {v['recommendation']}")
                lines.append("")

        lines.append("=" * 72)
        return "\n".join(lines)

    def _compute_grade_and_status(
        self,
        score: int,
        vulnerabilities: List[Dict[str, Any]],
        syntax_error: Optional[str] = None,
    ) -> Tuple[str, str]:
        high_cnt = sum(1 for v in vulnerabilities if v.get("severity") == "HIGH")

        if syntax_error:
            return "F", "ERROR"

        if score >= 95 and high_cnt == 0:
            grade = "A+"
        elif score >= 85 and high_cnt == 0:
            grade = "A"
        elif score >= 70:
            grade = "B"
        elif score >= 50:
            grade = "C"
        else:
            grade = "F"

        if score >= 85 and high_cnt == 0:
            status = "PASSED"
        elif score >= 50 and high_cnt == 0:
            status = "WARNING"
        else:
            status = "FAILED"

        return grade, status
