import argparse
import ast
import datetime
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
from typing import Any, Dict, List, Optional, Union

import torch

from chakra.agent import KimiAgent, LocalModelRunner
from chakra.multi_agent import MultiAgentOrchestrator
from chakra.model import K3Config, K3Model, tiny_config
from chakra.session import SessionManager
from chakra.persona import PersonaManager
from chakra.tokenizer import KimiTokenizer
from chakra.ui import (
    clear_screen,
    print_agent_step,
    print_banner,
    print_code_box,
    print_diff_box,
    print_sessions_list,
    print_step,
    print_vuln_report,
)

LOCAL_MODEL_DIR = Path("models/chakra_local")


def get_optimal_gen_tokens() -> int:
    """Get optimal gen_tokens from benchmark cache, or run benchmark if not available."""
    benchmark_file = Path(".chakra_benchmark.json")
    if benchmark_file.exists():
        try:
            data = json.loads(benchmark_file.read_text(encoding="utf-8"))
            return data.get("optimal_gen_tokens", 192)
        except Exception:
            pass
    return 192  # Default if no benchmark


def ensure_local_model() -> bool:
    """Checks if local model exists; downloads it if not. Returns True if model is ready."""
    if LOCAL_MODEL_DIR.exists() and any(LOCAL_MODEL_DIR.glob("*.safetensors")):
        return True
    if LOCAL_MODEL_DIR.exists() and any(LOCAL_MODEL_DIR.glob("*.bin")):
        return True
    if LOCAL_MODEL_DIR.exists() and (LOCAL_MODEL_DIR / "config.json").exists():
        return True

    print_step("MODEL", "Local model not found. Downloading Qwen2.5-Coder-1.5B...", "WAIT")
    try:
        import importlib.util
        script_path = Path(__file__).resolve().parent.parent / "tools" / "download_model.py"
        if script_path.exists():
            spec = importlib.util.spec_from_file_location("download_model", str(script_path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.download_model()
            return True
        else:
            print_step("MODEL", f"Download script not found at {script_path}", "WARN")
            return False
    except Exception as e:
        print_step("MODEL", f"Auto-download failed: {e}. Run 'python tools/download_model.py' manually.", "WARN")
        return False


PRESETS: Dict[str, Dict[str, Union[float, str, bool]]] = {
    "laptop": {
        "cache_gb": 0.5,
        "streaming_trunk": True,
        "description": "Optimized for 8GB RAM laptops (0.5GB expert cache, streamed trunk)",
    },
    "desktop": {
        "cache_gb": 4.0,
        "streaming_trunk": True,
        "description": "Optimized for 32GB RAM desktops (4GB expert cache, streamed trunk)",
    },
    "workstation": {
        "cache_gb": 32.0,
        "streaming_trunk": True,
        "description": "Optimized for 64GB-128GB RAM workstations (32GB expert cache, streamed trunk)",
    },
    "server": {
        "cache_gb": 128.0,
        "streaming_trunk": False,
        "description": "Optimized for high-RAM servers (128GB expert cache, resident trunk)",
    },
}

SESSIONS_DIR = Path(".chakra_sessions")


def audit_file_security(filepath: Union[str, Path]) -> List[Dict[str, Any]]:
    """Performs static InfoSec vulnerability and code security analysis on Python source file."""
    path = Path(filepath)
    if not path.exists() or not path.is_file():
        return [
            {
                "line": 0,
                "severity": "HIGH",
                "category": "File Error",
                "title": "File Not Found",
                "description": f"Target file does not exist: {filepath}",
                "remediation": "Provide a valid path to an existing file.",
            }
        ]

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return [
            {
                "line": 0,
                "severity": "HIGH",
                "category": "File Error",
                "title": "Read Error",
                "description": str(e),
                "remediation": "Check file permissions and path readability.",
            }
        ]

    findings: List[Dict[str, Any]] = []
    lines = content.splitlines()

    secret_pattern = re.compile(
        r"(?i)(api_key|password|secret|private_key|token|auth_token)\s*=\s*[\"'][A-Za-z0-9_\-~]{8,}[\"']"
    )
    eval_pattern = re.compile(r"\b(eval|exec)\s*\(")
    shell_pattern = re.compile(r"subprocess\.(call|Popen|run)\(.*shell\s*=\s*True.*\)")
    pickle_pattern = re.compile(r"\b(pickle\.loads|yaml\.unsafe_load)\b")
    md5_pattern = re.compile(r"\bhashlib\.(md5|sha1)\b")
    system_pattern = re.compile(r"\bos\.system\s*\(")

    for idx, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        if secret_pattern.search(line):
            findings.append(
                {
                    "line": idx,
                    "severity": "CRITICAL",
                    "category": "Credential Security",
                    "title": "Hardcoded Credential / Secret",
                    "description": f"Potential plain-text secret or key assignment: '{stripped[:40]}...'",
                    "remediation": "Store secrets in environment variables or key vault management.",
                }
            )
        if eval_pattern.search(line):
            findings.append(
                {
                    "line": idx,
                    "severity": "HIGH",
                    "category": "Code Injection",
                    "title": "Dynamic Code Execution (eval/exec)",
                    "description": f"Use of eval() or exec() allows arbitrary code execution: '{stripped[:40]}...'",
                    "remediation": "Avoid eval/exec; use safe parsing functions like ast.literal_eval.",
                }
            )
        if shell_pattern.search(line) or system_pattern.search(line):
            findings.append(
                {
                    "line": idx,
                    "severity": "HIGH",
                    "category": "Command Injection",
                    "title": "Unsafe Subprocess Shell Execution",
                    "description": f"Shell invocation detected (shell=True or os.system): '{stripped[:40]}...'",
                    "remediation": "Pass command arguments as a list without shell=True.",
                }
            )
        if pickle_pattern.search(line):
            findings.append(
                {
                    "line": idx,
                    "severity": "HIGH",
                    "category": "Insecure Deserialization",
                    "title": "Unsafe Object Deserialization",
                    "description": f"Unsafe pickle or YAML load: '{stripped[:40]}...'",
                    "remediation": "Use safe serialization like json or yaml.safe_load.",
                }
            )
        if md5_pattern.search(line):
            findings.append(
                {
                    "line": idx,
                    "severity": "MEDIUM",
                    "category": "Weak Cryptography",
                    "title": "Deprecated Hashing Algorithm",
                    "description": "MD5/SHA1 hash functions are cryptographically broken.",
                    "remediation": "Upgrade to SHA-256 or SHA-512 via hashlib.sha256.",
                }
            )

    try:
        tree = ast.parse(content, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                findings.append(
                    {
                        "line": getattr(node, "lineno", 0),
                        "severity": "LOW",
                        "category": "Code Robustness",
                        "title": "Assert Statement Usage",
                        "description": "Assert statements are stripped when Python runs in optimized mode (-O).",
                        "remediation": "Use explicit if-conditions and raise ValueError/TypeError for validation.",
                    }
                )
    except SyntaxError as se:
        findings.append(
            {
                "line": se.lineno or 0,
                "severity": "MEDIUM",
                "category": "Syntax Error",
                "title": "Python Syntax Error",
                "description": f"Failed to parse AST: {se.msg}",
                "remediation": "Fix python syntax error.",
            }
        )

    return findings


def generate_workspace_tree(dir_path: Path, max_depth: int = 3, current_depth: int = 0) -> str:
    """Generates formatted directory tree representation of the workspace."""
    if current_depth > max_depth:
        return ""

    lines: List[str] = []
    ignore_dirs = {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        ".pytest_cache",
        ".chakra_sessions",
        "egg-info",
        "dist",
        "build",
    }

    try:
        items = sorted(list(dir_path.iterdir()), key=lambda p: (not p.is_dir(), p.name.lower()))
    except Exception as e:
        return f"Error listing directory: {e}"

    for idx, item in enumerate(items):
        if item.name in ignore_dirs or (item.name.startswith(".") and item.name != "."):
            continue

        is_last = idx == len(items) - 1
        prefix = "└── " if is_last else "├── "
        indent = "│   " * current_depth

        if item.is_dir():
            lines.append(f"{indent}{prefix}{item.name}/")
            sub_tree = generate_workspace_tree(item, max_depth, current_depth + 1)
            if sub_tree:
                lines.append(sub_tree)
        else:
            lines.append(f"{indent}{prefix}{item.name}")

    return "\n".join(lines)


def get_workspace_context(root_dir: Path) -> Dict[str, Any]:
    """Indexes workspace Python files and computes total lines & size metadata."""
    ignore_dirs = {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        ".pytest_cache",
        ".chakra_sessions",
    }
    py_files: List[Dict[str, Any]] = []
    total_lines = 0

    for path in root_dir.rglob("*"):
        if any(part in ignore_dirs or part.startswith(".") for part in path.parts):
            continue
        if path.is_file() and path.suffix == ".py":
            try:
                lines_count = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
                size_bytes = path.stat().st_size
                py_files.append(
                    {
                        "path": str(path.relative_to(root_dir)),
                        "lines": lines_count,
                        "size": size_bytes,
                    }
                )
                total_lines += lines_count
            except Exception:
                pass

    return {
        "root": str(root_dir.resolve()),
        "total_py_files": len(py_files),
        "total_lines": total_lines,
        "files": py_files,
    }


def parse_prompt(prompt_str: str) -> List[int]:
    """Parses prompt input string into a list of token integer IDs."""
    if not prompt_str:
        return [1, 2, 3, 4, 5]

    parts = [p.strip() for p in prompt_str.split(",")]
    if all(p.isdigit() and p != "" for p in parts):
        return [int(p) for p in parts]

    return [ord(c) % 256 for c in prompt_str]


def run_repl(
    model: Any,
    tokenizer: KimiTokenizer,
    agent: KimiAgent,
    gen_tokens: int,
    device: str,
    incremental: bool,
) -> None:
    """Runs interactive Chakra-AI Agentic REPL shell for KimiPy."""
    orchestrator = MultiAgentOrchestrator(model=model, tokenizer=tokenizer, device=device, agent=agent)

    session_mgr = SessionManager(storage_dir=SESSIONS_DIR)
    persona_mgr = PersonaManager(initial_persona="fullstack")
    active_persona = persona_mgr._active_role
    active_session_id = f"sess_{int(time.time()) % 100000:05d}"
    previous_code: str = ""
    last_code: str = ""

    print_step("REPL", "Chakra-AI Agentic Terminal Active", "SUCCESS")
    print_step("PERSONA", f"Active Persona: [{active_persona.upper()}] - {persona_mgr.get_active_persona()['name']}", "INFO")
    print_step("SESSION", f"Active Session ID: {active_session_id}", "INFO")
    print_step("COMMANDS", "Type natural prompt or command to execute agent tasks.", "INFO")
    print_step(
        "SHORTCUTS",
        "/context | /tree | /audit <file> | /scan-vuln | /run <file> | /diff | /sessions | /resume <id> | /persona [role] | /help\n",
        "INFO",
    )

    while True:
        try:
            prompt_label = f"({active_persona}) > "
            user_input = input(prompt_label).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Chakra-AI REPL. Goodbye!")
            break

        if not user_input:
            continue

        # Exit commands
        if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
            session_mgr.save_session(
                session_id=active_session_id,
                history=agent.chat_history,
                metadata={"persona": active_persona, "last_code": agent.last_code or last_code},
            )
            print_step("REPL", f"Session '{active_session_id}' saved. Exiting Chakra-AI Terminal. Goodbye!", "INFO")
            break

        # Clear screen command
        if user_input.lower() == "/clear":
            clear_screen()
            print_banner()
            print_step("REPL", f"Screen cleared. Active Session: {active_session_id} | Persona: {active_persona}", "INFO")
            continue

        # Status command
        if user_input.lower() == "/status":
            print_step("STATUS", "Chakra-AI System Status", "INFO")
            # Engine info
            engine_name = type(model).__name__ if model else "None"
            if hasattr(model, "model_name"):
                engine_name = f"LocalModelRunner ({model.model_name})"
            elif hasattr(model, "config"):
                engine_name = f"K3Model ({model.config.num_hidden_layers}L/{model.config.hidden_size}H)"
            print(f"  Engine:      {engine_name}")
            print(f"  Device:      {device}")
            print(f"  Persona:     [{active_persona.upper()}] - {persona_mgr.get_active_persona()['name']}")
            print(f"  Session:     {active_session_id}")
            print(f"  Chat msgs:   {len(agent.chat_history)}")
            # Memory
            try:
                import psutil
                proc = psutil.Process()
                mem_mb = proc.memory_info().rss / (1024 * 1024)
                print(f"  RAM:         {mem_mb:.1f} MB")
            except ImportError:
                print("  RAM:         (install psutil for memory reporting)")
            # Model info
            if hasattr(model, "loaded"):
                print(f"  Model loaded: {model.loaded}")
            print()
            continue

        # Help command
        if user_input.lower() in ("/help", "/h", "?"):
            print_step("HELP", "Chakra-AI Agentic REPL Shell Commands:", "INFO")
            print("  <natural prompt>   - Generate, sandbox execute, and self-debug Python code with active persona")
            print("  /context           - Display workspace file index summary (files, line counts, sizes)")
            print("  /tree              - Render workspace visual directory tree structure")
            print("  /audit <filepath>  - Run InfoSec static security audit on a specific source file")
            print("  /scan-vuln         - Scan all Python files in workspace for security vulnerabilities")
            print("  /run <filepath>    - Directly execute a Python script file in sandbox runner")
            print("  /diff [file]       - Preview unified code diff between prior and latest code generations")
            print("  /sessions          - List all saved REPL chat & command sessions")
            print("  /resume <id>       - Resume/load a saved session by ID")
            print("  /persona [role]    - Switch persona (infosec, architect, devops, fullstack)")
            print("  /team <prompt>     - Launch Multi-Agent team collaboration mode")
            print("  /agents            - List active multi-agent team roles")
            print("  /save <filepath>   - Save last generated Python code block to a local file")
            print("  /clear             - Clear terminal screen")
            print("  /status            - Show engine info, memory usage, and session details")
            print("  /exit or /quit     - Save session and exit REPL shell\n")
            continue

        # Persona switcher command
        if user_input.startswith("/persona"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                active_info = persona_mgr.get_active_persona()
                print_step("PERSONA", f"Current Persona: [{active_persona.upper()}] - {active_info['name']}", "INFO")
                print_step("PERSONA", f"Available Roles: {', '.join(persona_mgr.list_personas().keys())}", "INFO")
                for r_key, r_val in persona_mgr.list_personas().items():
                    mark = " *" if r_key == active_persona else "  "
                    print(f"{mark} {r_key:<10} - {r_val['name']} ({r_val['description']})")
                print()
                continue
            role_arg = parts[1].strip().lower()
            try:
                persona_mgr.set_persona(role_arg)
                active_persona = persona_mgr._active_role
                print_step("PERSONA", f"Switched persona to: [{active_persona.upper()}] - {persona_mgr.get_active_persona()['name']}", "SUCCESS")
            except ValueError:
                print_step("PERSONA", f"Unknown role '{role_arg}'. Valid roles: {', '.join(persona_mgr.list_personas().keys())}", "WARN")
            continue

        # Context command
        if user_input.lower() == "/context":
            ctx = get_workspace_context(Path.cwd())
            print_step("CONTEXT", f"Workspace Root: {ctx['root']}", "INFO")
            print_step("CONTEXT", f"Total Python Files: {ctx['total_py_files']} | Total Lines: {ctx['total_lines']}", "INFO")
            for f in ctx["files"]:
                print(f"  • {f['path']:<35} | {f['lines']:>4} lines | {f['size']:>6} bytes")
            print()
            continue

        # Tree command
        if user_input.lower() == "/tree":
            print_step("TREE", f"Workspace Directory Tree ({Path.cwd().name}):", "INFO")
            tree_str = generate_workspace_tree(Path.cwd())
            if tree_str:
                print_code_box(tree_str, title=f"Tree: {Path.cwd().name}")
            else:
                print_step("TREE", "Workspace is empty.", "WARN")
            continue

        # Audit single file command
        if user_input.startswith("/audit"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                print_step("AUDIT", "Usage: /audit <filepath>", "WARN")
                continue
            audit_path = parts[1].strip()
            print_step("AUDIT", f"Running InfoSec security audit on: '{audit_path}'...", "WAIT")
            findings = audit_file_security(audit_path)
            print_vuln_report(audit_path, findings)
            continue

        # Scan vulnerability command
        if user_input.lower() == "/scan-vuln":
            print_step("SCAN", f"Scanning workspace '{Path.cwd().name}' for security vulnerabilities...", "WAIT")
            py_files = [p for p in Path.cwd().rglob("*.py") if not any(part.startswith(".") or part in (".venv", "__pycache__", "venv") for part in p.parts)]
            total_findings = 0
            if not py_files:
                print_step("SCAN", "No Python files found in workspace to audit.", "WARN")
            for py_file in py_files:
                rel_path = str(py_file.relative_to(Path.cwd()))
                findings = audit_file_security(py_file)
                if findings:
                    total_findings += len(findings)
                    print_vuln_report(rel_path, findings)
            if total_findings == 0:
                print_step("SCAN", f"Audit Complete across {len(py_files)} files: Zero security issues detected!", "SUCCESS")
            else:
                print_step("SCAN", f"Audit Complete: Found {total_findings} security finding(s) across workspace.", "WARN")
            print()
            continue

        # Direct file execution command
        if user_input.startswith("/run"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                print_step("RUN", "Usage: /run <filepath>", "WARN")
                continue
            run_path = parts[1].strip()
            print_step("RUN", f"Executing script file: '{run_path}' in sandbox...", "WAIT")
            exec_res = agent.run_file(run_path)
            if exec_res["success"]:
                print_step("RUN", f"Script execution SUCCESS (Exit Code: {exec_res['exit_code']})", "SUCCESS")
                if exec_res["stdout"].strip():
                    print_step("OUTPUT", exec_res["stdout"].strip(), "INFO")
            else:
                print_step("RUN", f"Script execution FAILED (Exit Code: {exec_res['exit_code']})", "FAIL")
                if exec_res["stderr"].strip():
                    print_step("ERROR", exec_res["stderr"].strip(), "FAIL")
            print()
            continue

        # Code Diff command
        if user_input.startswith("/diff"):
            parts = user_input.split(maxsplit=1)
            curr_code = agent.last_code or last_code
            if len(parts) > 1 and parts[1].strip():
                target_path = Path(parts[1].strip())
                if target_path.exists() and target_path.is_file():
                    old_c = target_path.read_text(encoding="utf-8", errors="replace")
                    diff_text = agent.generate_diff(old_c, curr_code, fromfile=str(target_path), tofile="Latest Generated Code")
                    print_diff_box(diff_text, title=f"Diff: {target_path.name} vs Latest")
                else:
                    print_step("DIFF", f"File not found: {target_path}", "FAIL")
            else:
                if previous_code or curr_code:
                    diff_text = agent.generate_diff(previous_code, curr_code, fromfile="Previous Attempt", tofile="Latest Generated Code")
                    print_diff_box(diff_text, title="Generated Code Diff Preview")
                else:
                    print_step("DIFF", "No prior code iterations available to compare.", "WARN")
            continue

        # Sessions list command
        if user_input.lower() == "/sessions":
            saved_sessions = session_mgr.list_sessions()
            # Adapt to print_sessions_list expected format
            adapted = []
            for s in saved_sessions:
                adapted.append({
                    "id": s.get("session_id", "unknown"),
                    "title": s.get("metadata", {}).get("title", "REPL Session"),
                    "timestamp": s.get("updated_at", ""),
                    "message_count": s.get("history_count", 0),
                })
            print_sessions_list(adapted, active_session_id=active_session_id)
            continue

        # Resume session command
        if user_input.startswith("/resume"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                print_step("RESUME", "Usage: /resume <session_id>", "WARN")
                continue
            res_id = parts[1].strip()
            try:
                session_data = session_mgr.load_session(res_id)
                active_session_id = session_data.get("session_id", res_id)
                active_persona = session_data.get("metadata", {}).get("persona", "fullstack")
                try:
                    persona_mgr.set_persona(active_persona)
                except ValueError:
                    active_persona = persona_mgr._active_role
                agent.chat_history = session_data.get("history", [])
                agent.last_code = session_data.get("metadata", {}).get("last_code", "")
                last_code = agent.last_code or ""
                print_step("RESUME", f"Successfully resumed session '{res_id}'", "SUCCESS")
                print_step("RESUME", f"Restored persona: [{active_persona.upper()}] | History: {len(agent.chat_history)} message(s)", "INFO")
            except FileNotFoundError:
                print_step("RESUME", f"Session '{res_id}' not found. Use /sessions to list available IDs.", "FAIL")
            continue

        # List active team agents command
        if user_input.lower() in ("/agents", "/roles"):
            print_step("AGENTS", "Active Multi-Agent Team Roles:", "INFO")
            for role, desc in orchestrator.list_agents().items():
                print_agent_step(role, desc)
            print()
            continue

        # Save code command
        if user_input.startswith("/save"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                print_step("SAVE", "Usage: /save <filepath>", "WARN")
                continue
            save_path = parts[1].strip()
            code_to_save = agent.last_code or last_code
            if not code_to_save:
                print_step("SAVE", "No generated code snippet is available to save.", "FAIL")
                continue
            try:
                out_path = Path(save_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(code_to_save, encoding="utf-8")
                print_step("SAVE", f"Code successfully saved to: {out_path.resolve()}", "SUCCESS")
            except Exception as err:
                print_step("SAVE", f"Failed to save file: {err}", "FAIL")
            continue

        # Team collaboration command
        if user_input.startswith("/team"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                print_step("TEAM", "Usage: /team <prompt>", "WARN")
                continue
            team_prompt = parts[1].strip()
            persona_prefix = persona_mgr.get_system_prompt()
            full_prompt = f"{persona_prefix}\n{team_prompt}"
            print_step("TEAM", f"[{active_persona.upper()}] Launching Team Collaboration: '{team_prompt}'...", "WAIT")

            res = orchestrator.run_team_collaboration(
                prompt=full_prompt,
                max_retries=3,
                gen_tokens=gen_tokens,
                incremental=incremental,
            )

            previous_code = last_code
            last_code = res.get("code", "")
            if last_code:
                print_code_box(last_code, title=f"TEAM GENERATED CODE [{active_persona.upper()}]")

            if res.get("success", False):
                print_step("SANDBOX", f"Execution SUCCESS (Completed in {res.get('iterations', 1)} iteration(s))", "SUCCESS")
                if res.get("stdout", "").strip():
                    print_step("OUTPUT", res["stdout"].strip(), "INFO")
            else:
                print_step("SANDBOX", f"Execution COMPLETED with warnings after {res.get('iterations', 1)} iteration(s)", "WARN")
                if res.get("stderr", "").strip():
                    print_step("ERROR", res["stderr"].strip(), "FAIL")

            session_mgr.save_session(
                session_id=active_session_id,
                history=agent.chat_history,
                metadata={"persona": active_persona, "last_code": agent.last_code or last_code},
            )
            print()
            continue

        # Determine intent: Code/Task Generation vs Conversational Chat
        task_keywords = {
            "code", "script", "program", "task", "create", "make", "write", "build", "generate",
            "calculator", "calendar", "scanner", "hash", "folder", "directory", "file",
            "function", "class", "module", "api", "app", "fix", "debug", "refactor", "run", "audit"
        }
        is_code_task = user_input.startswith("/code") or any(k in user_input.lower() for k in task_keywords)

        persona_prefix = persona_mgr.get_system_prompt()
        full_prompt = f"{persona_prefix}\n{user_input}"

        if is_code_task:
            code_prompt = user_input[5:].strip() if user_input.startswith("/code") else user_input
            if not code_prompt:
                print_step("AGENT", "Usage: /code <python task prompt>", "WARN")
                continue

            print_step("AGENT", f"[{active_persona.upper()}] Running task: '{code_prompt}'...", "WAIT")
            print_step("AGENT", f"Generating code ({gen_tokens} tokens on CPU, please wait)...", "INFO")
            res = agent.run_agentic_loop(
                prompt=full_prompt,
                max_retries=3,
                gen_tokens=gen_tokens,
                incremental=incremental,
            )

            previous_code = last_code
            last_code = res["code"]

            # Auto-save code to file instead of displaying
            try:
                output_dir = Path("chakra_output")
                output_dir.mkdir(exist_ok=True)
                code_file = output_dir / "generated_script.py"
                code_file.write_text(last_code, encoding="utf-8")
                print_step("SAVE", f"Code saved to: {code_file.resolve()}", "INFO")

                # Execute the saved file
                exec_res = agent.run_in_sandbox(code_file, timeout=30)
                if exec_res["success"]:
                    print_step("DONE", "Task completed successfully", "SUCCESS")
                    if exec_res["stdout"].strip():
                        print_step("OUTPUT", exec_res["stdout"].strip(), "INFO")
                else:
                    print_step("DONE", "Task completed with errors", "WARN")
                    if exec_res["stderr"].strip():
                        print_step("ERROR", exec_res["stderr"].strip(), "FAIL")
            except Exception as err:
                print_step("ERROR", f"Failed to save/execute: {err}", "FAIL")

            if res["success"]:
                print_step("SANDBOX", f"Execution SUCCESS (Completed in {res['iterations']} iteration(s))", "SUCCESS")
            else:
                print_step("SANDBOX", f"Execution FAILED after {res['iterations']} iteration(s)", "FAIL")
        else:
            chat_reply = agent.chat(full_prompt, gen_tokens=gen_tokens, incremental=incremental)
            print_step("ASSISTANT", chat_reply, "SUCCESS")

        session_mgr.save_session(
            session_id=active_session_id,
            history=agent.chat_history,
            metadata={"persona": active_persona, "last_code": agent.last_code or last_code},
        )
        print()



def main(args: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="chakra: Pure PyTorch runner and CLI for Kimi K3"
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default="laptop",
        help="Hardware preset profile (default: laptop)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Input text prompt or comma-separated token IDs",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Launch interactive REPL chat mode",
    )
    parser.add_argument(
        "--gen",
        type=int,
        default=192,
        help="Number of tokens to generate (default: 192)",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        default=True,
        help="Use stateful incremental decoding (default: True)",
    )
    parser.add_argument(
        "--no-incremental",
        action="store_false",
        dest="incremental",
        help="Disable incremental decoding and re-forward full sequence each step",
    )
    parser.add_argument(
        "--trunk",
        type=str,
        default=None,
        help="Path to trunk.bin for Option A (full 93-layer Kimi K3 MoE). Without this, Option B (lightweight local model) is used.",
    )
    parser.add_argument(
        "--local-model",
        type=str,
        default=None,
        help="Path to local model directory for Option B (default: models/chakra_local/)",
    )
    parser.add_argument(
        "--cache-gb",
        type=float,
        default=None,
        help="Expert cache size limit in GB (overrides preset value)",
    )
    parser.add_argument(
        "--trunk-gb",
        type=float,
        default=None,
        help="Trunk memory budget in GB for ring buffer streaming (default: auto from preset)",
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "c-backend", "pytorch", "local"],
        default="auto",
        help="Inference engine: c-backend (kimi-k3-in-c), pytorch (model.py), local (Qwen2.5), auto (detect best)",
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Run tiny synthetic model (13 layers) for fast verification",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Target device (cpu or cuda)",
    )

    parsed = parser.parse_args(args)

    preset_info = PRESETS[parsed.preset]
    cache_gb = parsed.cache_gb if parsed.cache_gb is not None else preset_info["cache_gb"]

    print_banner()
    print_step("CONFIG", f"Preset: {parsed.preset} ({preset_info['description']})", "INFO")
    print_step("CONFIG", f"Expert Cache: {cache_gb:.2f} GB | Incremental: {parsed.incremental} | Device: {parsed.device}", "INFO")

    # Determine engine: Option A (--trunk) vs Option B (local model, default)
    use_option_a = parsed.trunk is not None
    use_option_b = not use_option_a and not parsed.tiny

    model = None
    tokenizer = KimiTokenizer(mode="auto")

    if use_option_a:
        # Option A: Full 93-layer Kimi K3 MoE with trunk streaming
        print_step("ENGINE", "Option A: Full 93-Layer Kimi K3 MoE (Trunk Streaming)", "INFO")
        print_step("CONFIG", f"Trunk Path: {parsed.trunk}", "INFO")
        cfg = K3Config()
        print_step("MODEL", f"Full Kimi K3 ({cfg.num_hidden_layers} layers, hidden={cfg.hidden_size})", "INFO")
        model = K3Model(cfg).to(parsed.device)
        model.eval()

    elif use_option_b:
        # Option B: Lightweight local trained model (default)
        local_path = Path(parsed.local_model) if parsed.local_model else LOCAL_MODEL_DIR
        print_step("ENGINE", "Option B: Lightweight Local Trained Model (~2.5 GB RAM)", "INFO")

        # Auto-download if model not present
        if not local_path.exists() or not any(local_path.glob("*")):
            print_step("MODEL", "Local model not found. Attempting download...", "WAIT")
            ensure_local_model()

        if local_path.exists():
            print_step("MODEL", f"Loading local model from: {local_path}", "WAIT")
            runner = LocalModelRunner(model_path=str(local_path), device=parsed.device)
            if runner.loaded:
                model = runner
                print_step("MODEL", f"Local model loaded: {runner.model_name}", "SUCCESS")
            else:
                print_step("MODEL", "Local model load failed. Falling back to echo mode.", "WARN")
        else:
            print_step("MODEL", "No local model available. Using echo fallback mode.", "WARN")

    elif parsed.tiny:
        # Tiny synthetic mode for verification
        cfg = tiny_config()
        print_step("ENGINE", "Tiny Synthetic Model (13 layers, verification mode)", "INFO")
        print_step("MODEL", f"Tiny Config ({cfg.num_hidden_layers} layers, hidden={cfg.hidden_size})", "INFO")
        model = K3Model(cfg).to(parsed.device)
        model.eval()

    # Initialize agent (auto-detects local model if model is None)
    agent = KimiAgent(model=model, tokenizer=tokenizer, device=parsed.device)
    if model is None:
        model = agent.model

    # Determine if REPL mode should be entered (default if no prompt provided, or if --chat set)
    if parsed.chat or parsed.prompt is None:
        run_repl(
            model=model,
            tokenizer=tokenizer,
            agent=agent,
            gen_tokens=parsed.gen,
            device=parsed.device,
            incremental=parsed.incremental,
        )
        return 0

    # Single Prompt Mode execution
    prompt_str = parsed.prompt

    print_step("PROMPT", f"Input: '{prompt_str}'", "INFO")
    print_step("GENERATE", f"Generating {parsed.gen} tokens...", "WAIT")

    # Route through LocalModelRunner's native tokenizer when available
    if isinstance(model, LocalModelRunner):
        t0 = time.time()
        decoded_text = model.generate(prompt_str, n_new=parsed.gen)
        el = time.time() - t0
        print_step("RESULT", f"Output:\n{decoded_text}", "SUCCESS")
        print_step("BENCH", f"Completed in {el:.3f} s", "INFO")
        return 0

    # Fallback: use KimiTokenizer + tensor path for K3Model/tiny
    if any(c.isalpha() for c in prompt_str):
        prompt_ids = tokenizer.encode(prompt_str)
    else:
        prompt_ids = parse_prompt(prompt_str)

    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=parsed.device)

    print_step("PROMPT", f"Tokens: {prompt_ids}", "INFO")

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            n_new=parsed.gen,
            incremental=parsed.incremental,
        )
    el = time.time() - t0

    generated_tokens = output_ids[0].tolist()
    new_tokens = generated_tokens[len(prompt_ids) :]
    decoded_text = tokenizer.decode(new_tokens)
    safe_decoded = decoded_text.encode("ascii", errors="backslashreplace").decode("ascii")

    print_step("RESULT", f"Full Sequence: {generated_tokens}", "INFO")
    print_step("RESULT", f"New Tokens: {new_tokens}", "INFO")
    out_repr = repr(decoded_text) if sys.stdout.encoding and sys.stdout.encoding.lower() == 'utf-8' else repr(safe_decoded)
    print_step("RESULT", f"Decoded Text: {out_repr}", "SUCCESS")
    print_step("BENCH", f"Completed in {el:.3f} s ({parsed.gen / max(el, 1e-9):.2f} tokens/s)", "INFO")

    return 0


if __name__ == "__main__":
    sys.exit(main())
