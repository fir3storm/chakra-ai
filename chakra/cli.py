import argparse
import ast
import datetime
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Union


from chakra.agent import KimiAgent, LocalModelRunner
from chakra.engine_llama import LlamaCppBackend
from chakra.multi_agent import MultiAgentOrchestrator
from chakra.model import K3Config, K3Model, tiny_config
from chakra.session import SessionManager
from chakra.persona import PersonaManager
from chakra.tokenizer import KimiTokenizer
from chakra.updater import check_and_notify
from chakra.ui import (
    ProgressBar,
    Spinner,
    clear_screen,
    print_agent_step,
    print_banner,
    print_chat_role,
    print_code_box,
    print_diff_box,
    print_header,
    print_patch_chunks,
    print_sessions_list,
    print_step,
    print_tool,
    print_vuln_report,
)
from chakra.security import InfoSecAuditor
from chakra.workspace import WorkspaceIndexer, scan_ast_symbols

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

    # System context for accurate responses
    now = datetime.datetime.now()
    system_info = f"{now.strftime('%A, %B %d, %Y %H:%M')} | {sys.platform}"
    print_step("SYSTEM", system_info, "INFO")

    # Load project memory from .chakra_memory
    memory_file = Path(".chakra_memory")
    project_memory = ""
    if memory_file.exists():
        try:
            project_memory = memory_file.read_text(encoding="utf-8").strip()
            if project_memory:
                print_step("MEMORY", f"Loaded {len(project_memory.splitlines())} lines of project context", "INFO")
        except Exception:
            pass

    # Workspace summary for conversation memory
    try:
        py_files = sorted(Path.cwd().glob("*.py"))
        if py_files:
            file_list = ", ".join(f"{f.name} ({len(f.read_text(encoding='utf-8', errors='replace').splitlines())}L)" for f in py_files[:15])
            print_step("WORKSPACE", f"{Path.cwd().name}: {file_list}", "INFO")
    except Exception:
        pass

    print_step("COMMANDS", "Type natural prompt or command to execute agent tasks.", "INFO")
    print_step(
        "SHORTCUTS",
            "/context | /tree | /audit <file> | /scan-vuln | /run <file> | /edit <file> <instruction> | /diff | /sessions | /resume <id> | /persona [role] | /help\n",
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
            print("  /edit <file> <cmd> - Edit a file with AI assistance")
            print("  /diff [file]       - Preview unified code diff between prior and latest code generations")
            print("  /sessions          - List all saved REPL chat & command sessions")
            print("  /resume <id>       - Resume/load a saved session by ID")
            print("  /persona [role]    - Switch persona (infosec, architect, devops, fullstack)")
            print("  /team <prompt>     - Launch Multi-Agent team collaboration mode")
            print("  /plan <task>       - Multi-step task planning & execution")
            print("  /edit <file> <cmd> - Edit a file with AI assistance")
            print("  /git [cmd]         - Run git commands (status, diff, commit)")
            print("  /memory [text]     - View/save project context for future sessions")
            print("  /agents            - List active multi-agent team roles")
            print("  /tools             - List tools available to Coder/Architect/Auditor agents")
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

        # Auto-remediate security issues command
        if user_input.startswith("/fix-sec"):
            parts = user_input.split(maxsplit=1)
            target = parts[1].strip() if len(parts) > 1 else None
            auditor = InfoSecAuditor()
            if target and Path(target).exists():
                print_step("FIX-SEC", f"Auditing and auto-remediating security issues in '{target}'...", "WAIT")
                res = auditor.auto_remediate_file(target)
                print_step("FIX-SEC", res.get("status", "Completed"), "SUCCESS" if res.get("fixes_applied", 0) > 0 else "INFO")
            else:
                print_step("FIX-SEC", "Auditing workspace files for security auto-remediation...", "WAIT")
                py_files = [p for p in Path.cwd().rglob("*.py") if not any(part.startswith(".") or part in (".venv", "__pycache__", "venv") for part in p.parts)]
                fixed_count = 0
                for pf in py_files:
                    res = auditor.auto_remediate_file(pf)
                    if res.get("fixes_applied", 0) > 0:
                        fixed_count += res["fixes_applied"]
                        print_step("FIX-SEC", f"Fixed {res['fixes_applied']} issue(s) in {pf.name}", "SUCCESS")
                if fixed_count == 0:
                    print_step("FIX-SEC", "No auto-fixable security issues found across workspace.", "INFO")
                else:
                    print_step("FIX-SEC", f"Total security fixes applied across workspace: {fixed_count}", "SUCCESS")
            print()
            continue

        # Automated testing and fixing loop command
        if user_input.startswith("/test"):
            parts = user_input.split(maxsplit=1)
            arg = parts[1].strip() if len(parts) > 1 else "pytest"
            test_cmd = "pytest"
            target_file = None
            if arg.endswith(".py"):
                target_file = arg
                test_cmd = f"pytest {arg}"
            else:
                test_cmd = arg

            print_step("TEST", f"Running automated test loop: '{test_cmd}'...", "WAIT")
            test_res = agent.test_and_fix(test_cmd=test_cmd, target_file=target_file)
            if test_res["success"]:
                print_step("TEST", test_res["message"], "SUCCESS")
            else:
                print_step("TEST", test_res["message"], "FAIL")
            print()
            continue

        # Interactive patch preview command
        if user_input.startswith("/patch"):
            parts = user_input.split(maxsplit=2)
            if len(parts) < 3:
                print_step("PATCH", "Usage: /patch <file> <instruction>", "WARN")
                continue
            target_file, patch_inst = parts[1].strip(), parts[2].strip()
            target_path = Path(target_file)
            if not target_path.exists():
                print_step("PATCH", f"File not found: {target_file}", "FAIL")
                continue
            old_code = target_path.read_text(encoding="utf-8", errors="replace")
            print_step("PATCH", f"Generating patch for {target_file} based on instruction: '{patch_inst}'...", "WAIT")
            prompt = f"Modify the following python file ({target_file}) according to this request: {patch_inst}\n\nOriginal Code:\n```python\n{old_code}\n```\n\nReturn ONLY the modified complete python code."
            new_code_raw = str(agent.model(prompt)) if agent.model else old_code
            new_code = agent.extract_code(new_code_raw)
            diff_text = agent.generate_diff(old_code, new_code, fromfile=f"a/{target_file}", tofile=f"b/{target_file}")
            print_patch_chunks(target_file, diff_text)

            confirm = input("Apply this patch? [y/N]: ").strip().lower()
            if confirm in ("y", "yes"):
                target_path.write_text(new_code, encoding="utf-8")
                print_step("PATCH", f"Patch applied successfully to {target_file}!", "SUCCESS")
            else:
                print_step("PATCH", "Patch discarded.", "INFO")
            print()
            continue

        # Codebase AST Symbol Graph command
        if user_input.lower() in ("/ast", "/symbols"):
            print_step("AST", f"Building AST Symbol Graph for '{Path.cwd().name}'...", "WAIT")
            indexer = WorkspaceIndexer(root_dir=Path.cwd())
            graph = scan_ast_symbols(indexer)
            print_code_box(graph.summary(), title=f"AST Symbol Graph ({Path.cwd().name})")
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

        # ── /tools command: list the tool-calling registry ──
        if user_input.lower() == "/tools":
            from chakra.tools import build_default_tools

            registry = build_default_tools(workspace_root=str(Path.cwd()), kimi_agent=agent)
            print_step("TOOLS", f"{len(registry.names())} tool(s) available to Coder/Architect/Auditor agents:", "INFO")
            for line in registry.describe().splitlines():
                print(f"  {line}")
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
            now = datetime.now()
            date_context = f"Current date: {now.strftime('%A, %B %d, %Y %H:%M')}. OS: {sys.platform}."
            persona_prefix = persona_mgr.get_system_prompt()
            full_prompt = f"{date_context}\n{persona_prefix}\n{team_prompt}"
            print_step("TEAM", f"[{active_persona.upper()}] Launching Team Collaboration: '{team_prompt}'...", "WAIT")

            res = orchestrator.run_team_collaboration(
                prompt=full_prompt,
                max_retries=3,
                gen_tokens=gen_tokens,
                incremental=incremental,
            )

            previous_code = last_code
            last_code = res.get("code", "")
            # Save team output to file instead of displaying
            output_dir = Path("chakra_output")
            output_dir.mkdir(exist_ok=True)
            team_file = output_dir / f"team_output_{active_session_id}.py"
            team_file.write_text(last_code, encoding="utf-8")

            if res.get("success", False):
                print_step("SANDBOX", f"Execution SUCCESS (Completed in {res.get('iterations', 1)} iteration(s))", "SUCCESS")
                if res.get("stdout", "").strip():
                    print_step("OUTPUT", res["stdout"].strip(), "INFO")
            else:
                print_step("SANDBOX", f"Execution COMPLETED with warnings after {res.get('iterations', 1)} iteration(s)", "WARN")
                if res.get("stderr", "").strip():
                    print_step("ERROR", res["stderr"].strip(), "FAIL")

            # Tool-call trace: what the Architect/Coder actually did via tools this run
            # (only populated when the active engine supports grammar-constrained tool calling).
            tool_log = list(getattr(orchestrator.architect, "tool_call_log", [])) + \
                list(getattr(orchestrator.coder, "tool_call_log", []))
            if tool_log:
                print_step("TOOLS", f"{len(tool_log)} tool call(s) made during collaboration:", "INFO")
                for tc in tool_log:
                    ok = tc.get("result", {}).get("success", False)
                    mark = "OK" if ok else "FAIL"
                    print_agent_step(tc.get("role", "Agent"), f"[{mark}] {tc.get('tool')}({tc.get('args')})")

            session_mgr.save_session(
                session_id=active_session_id,
                history=agent.chat_history,
                metadata={"persona": active_persona, "last_code": agent.last_code or last_code},
            )
            print()
            continue

        # Determine intent: Code/Task Generation vs Conversational Chat

        # ── /plan command: multi-step task planning ──
        if user_input.startswith("/plan"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print_step("PLAN", "Usage: /plan <complex task description>", "WARN")
                continue

            plan_prompt = parts[1].strip()
            print_header(f"PLAN: {plan_prompt}")
            persona_prefix = persona_mgr.get_system_prompt()
            date_context = f"Current date: {now.strftime('%A, %B %d, %Y %H:%M')}. OS: {sys.platform}."

            # Step 1: Generate the plan
            print_step("PLAN", "Step 1: Generating task breakdown...", "WAIT")
            breakdown_prompt = f"{date_context}\n{persona_prefix}\nBreak down this complex request into numbered steps (max 5). Output ONLY the step list, one per line, starting with '1.', no extra text:\n{plan_prompt}"

            try:
                if isinstance(model, LlamaCppBackend) and model.loaded:
                    plan_text = model.generate(breakdown_prompt, n_new=128)
                elif isinstance(model, LocalModelRunner) and model.loaded:
                    plan_text = model.generate(breakdown_prompt, n_new=128)
                else:
                    plan_text = agent.chat(breakdown_prompt, gen_tokens=128)
            except Exception:
                plan_text = "1. Implement the core logic\n2. Add error handling\n3. Write the main entry point"

            # Parse steps
            steps = []
            for line in plan_text.splitlines():
                line = line.strip()
                if line and (line[0].isdigit() and '.' in line[:3]):
                    step_text = line.split('.', 1)[1].strip()
                    if step_text:
                        steps.append(step_text)
            if not steps:
                steps = [plan_prompt]

            print_step("PLAN", f"Found {len(steps)} step(s):", "INFO")
            for i, s in enumerate(steps, 1):
                print(f"  {i}. {s}")

            # Step 2: Execute each step
            results = []
            for i, step_desc in enumerate(steps, 1):
                print_header(f"Step {i}/{len(steps)}: {step_desc[:60]}")
                with Spinner(f"Executing step {i}...") as sp:
                    step_prompt = f"{date_context}\n{persona_prefix}\nWrite a complete Python script for this specific task only:\n{step_desc}\nEnclose in ```python ... ```"
                    res = agent.run_agentic_loop(prompt=step_prompt, max_retries=1, gen_tokens=gen_tokens)
                    results.append({"step": i, "task": step_desc, "code": res.get("code", ""), "success": res.get("success", False)})
                    sp.set_result(f"{'OK' if res.get('success') else 'FAIL'}")
                    print_tool("execute" if res.get("success") else "error", f"Step {i} complete")

            # Step 3: Save .chakra_memory
            mem_lines = [f"Task: {plan_prompt}"]
            for r in results:
                mem_lines.append(f"  Step {r['step']}: {r['task']} ({'OK' if r['success'] else 'FAIL'})")
            Path(".chakra_memory").write_text("\n".join(mem_lines), encoding="utf-8")
            project_memory = "\n".join(mem_lines)
            print_tool("save", ".chakra_memory", f"→ {len(mem_lines)} lines")
            print_step("PLAN", f"All {len(steps)} steps completed. Memory saved.", "SUCCESS")
            continue

        # ── /git command ──
        if user_input.startswith("/git"):
            parts = user_input.split(maxsplit=1)
            git_cmd = parts[1].strip() if len(parts) > 1 else "status"
            import subprocess as sp
            try:
                result = sp.run(["git"] + git_cmd.split(), capture_output=True, text=True, cwd=str(Path.cwd()), timeout=15)
                if result.returncode == 0:
                    print_step("GIT", f"git {git_cmd} — OK", "SUCCESS")
                    if result.stdout.strip():
                        print(result.stdout.strip())
                else:
                    print_step("GIT", f"git {git_cmd} failed", "FAIL")
                    if result.stderr.strip():
                        print(result.stderr.strip()[:500])
            except FileNotFoundError:
                print_step("GIT", "git not installed or not in PATH", "WARN")
            except sp.TimeoutExpired:
                print_step("GIT", "git command timed out", "WARN")
            except Exception as e:
                print_step("GIT", f"Error: {e}", "FAIL")
            continue

        # ── /memory command ──
        if user_input.startswith("/memory"):
            parts = user_input.split(maxsplit=1)
            if len(parts) > 1 and parts[1].strip():
                new_memory = parts[1].strip()
                Path(".chakra_memory").write_text(new_memory, encoding="utf-8")
                project_memory = new_memory
                print_tool("save", ".chakra_memory", f"→ {len(new_memory.splitlines())} lines")
            else:
                if project_memory:
                    print_step("MEMORY", f"\n{project_memory}", "INFO")
                else:
                    print_step("MEMORY", "No project memory set. Use /memory <text> to save context.", "INFO")
            continue

        # ── /edit command ──
        # Determine intent: Code/Task Generation vs Conversational Chat
        # Check for /edit command first
        if user_input.startswith("/edit"):
            parts = user_input.split(maxsplit=2)
            if len(parts) < 3:
                print_step("EDIT", "Usage: /edit <filepath> <instruction>", "WARN")
                continue
            target_file = Path(parts[1])
            instruction = parts[2]

            if not target_file.exists():
                print_step("EDIT", f"File not found: {target_file}", "FAIL")
                continue

            existing_code = target_file.read_text(encoding="utf-8", errors="replace")
            edit_prompt = f"Modify this Python file according to the instruction.\nInstruction: {instruction}\n\nExisting code:\n{existing_code}\n\nOutput the complete modified file in ```python ... ``` block."

            with Spinner(f"Editing {target_file.name}...") as sp:
                res = agent.run_agentic_loop(
                    prompt=edit_prompt,
                    max_retries=1,
                    gen_tokens=gen_tokens,
                    incremental=incremental,
                )
                sp.set_result(f"{len(res['code'].splitlines())} lines")

            if res["code"] and res["code"] != existing_code:
                diff = agent.generate_diff(existing_code, res["code"], fromfile=str(target_file), tofile=f"{target_file} (edited)")
                print_diff_box(diff, title=f"Preview: {target_file.name}")
                ask = input(f"Apply changes to {target_file}? [y/N] ").strip().lower()
                if ask == "y":
                    target_file.write_text(res["code"], encoding="utf-8")
                    print_tool("save", str(target_file), f"→ {len(res['code'].splitlines())} lines")
                    print_step("EDIT", f"Changes applied to {target_file}", "SUCCESS")
                else:
                    print_step("EDIT", "Edit discarded.", "INFO")
            else:
                print_step("EDIT", "No changes generated.", "WARN")
            continue

        task_keywords = {
            "create ", "make ", "write ", "build ", "generate ",
            "calculator", "calendar", "scanner ", "hash ", "folder ", "directory ",
            "function ", "class ", "module ", "api ", "app ", "refactor ",
        }
        is_code_task = user_input.startswith("/code") or any(k in user_input.lower() for k in task_keywords)

        persona_prefix = persona_mgr.get_system_prompt()
        date_context = f"Current date: {now.strftime('%A, %B %d, %Y %H:%M')}. OS: {sys.platform}."
        memory_context = f"\nProject knowledge: {project_memory}" if project_memory else ""
        # Build system message — never echoed by the model
        system_msg = f"{date_context}\n{persona_prefix}{memory_context}"
        full_prompt = user_input  # Clean user message only
        ws_context = ""  # filled below for code tasks

        if is_code_task:
            code_prompt = user_input[5:].strip() if user_input.startswith("/code") else user_input
            if not code_prompt:
                print_step("AGENT", "Usage: /code <python task prompt>", "WARN")
                continue

            # Inject workspace context
            workspace_files = []
            for f in Path.cwd().glob("*.py"):
                if f.stat().st_size < 10000:
                    try:
                        f.read_text(encoding="utf-8")
                        workspace_files.append(f"{f.name}")
                    except Exception:
                        pass
            ws_context = ""
            if workspace_files:
                ws_context = f"\nWorking directory: {Path.cwd()}\nExisting files: {', '.join(workspace_files)}\n"

            # Detect target language/filename from user prompt
            target_ext = ".py"
            target_lang = "Python"
            for word in code_prompt.lower().replace(".", " ").replace(",", " ").split():
                if word in ("php", ".php"):
                    target_ext = ".php"
                    target_lang = "PHP"
                elif word in ("html", ".html"):
                    target_ext = ".html"
                    target_lang = "HTML"
                elif word in ("js", "javascript", ".js"):
                    target_ext = ".js"
                    target_lang = "JavaScript"
                elif word in ("css", ".css"):
                    target_ext = ".css"
                    target_lang = "CSS"
                elif word in ("sh", "bash", "shell", ".sh"):
                    target_ext = ".sh"
                    target_lang = "Bash"

            # Detect output filename from prompt
            output_filename = None
            for word in code_prompt.replace("=", " ").replace(":", " ").replace('"', ' ').replace("'", " ").split():
                if "." in word and len(word) > 2 and word.count(".") == 1:
                    ext = word.split(".")[-1].lower()
                    if ext in ("py", "php", "html", "js", "css", "sh", "json", "yaml", "md", "txt", "bat", "ps1"):
                        output_filename = word
                        target_ext = f".{ext}"
                        target_lang = ext.upper()
                        break

            # Clean prompt — detect and respect target language
            if target_lang != "Python":
                code_fence = f"```{target_ext[1:]}"
                clean_code_prompt = f"You are a {target_lang} expert. Write a complete, production-ready {target_lang} file.\n\nRequest: {code_prompt}\n\nWorkspace: {ws_context}\n\nIMPORTANT: Start your response with {code_fence} and end with ```. Only output code inside the code fence."
            else:
                clean_code_prompt = (
                    f"Write a complete, production-ready Python application for the following request:\n"
                    f"{ws_context}{code_prompt}\n\n"
                    f"CRITICAL REQUIREMENTS:\n"
                    f"1. If the user asks for a GUI app (Tkinter/PyQt), write the COMPLETE runnable script including all event handlers, "
                    f"button callbacks, grid layout, styling, and `root.mainloop()` at the end so it opens a GUI window immediately when executed!\n"
                    f"2. If the user asks for a CLI tool (calculator, calendar, manager, etc.), build a FULLY INTERACTIVE REPL app with a main loop (`while True:`), "
                    f"interactive user input (`input(...)`), formatted menu, robust error handling, and clean output — DO NOT generate static hardcoded print statements!\n"
                    f"3. Always enclose the complete python code inside a ```python ... ``` code block."
                )

            code_gen_tokens = max(gen_tokens, 768)
            pbar = ProgressBar(title=f"Building {target_lang} App", total_passes=4)

            def progress_cb(current_pass: int, new_tokens: int, status: str):
                pbar.update(current_pass=current_pass, new_tokens=new_tokens, status=status)

            res = agent.run_agentic_loop(
                prompt=clean_code_prompt,
                max_retries=1 if target_ext != ".py" else 3,
                gen_tokens=code_gen_tokens,
                incremental=incremental,
                progress_callback=progress_cb,
            )
            pbar.finish(message=f"Generated {len(res['code'].splitlines())} lines of code")

            previous_code = last_code
            last_code = res["code"]

            # Auto-save with correct extension + skip execution for non-Python
            try:
                output_dir = Path("chakra_output")
                output_dir.mkdir(exist_ok=True)
                if output_filename:
                    code_file = output_dir / output_filename
                else:
                    code_file = output_dir / f"generated_script{target_ext}"
                code_file.write_text(last_code, encoding="utf-8")
                print_tool("save", str(code_file), f"→ {len(last_code.splitlines())} lines")

                # Only sandbox execute Python files
                if target_ext == ".py":
                    exec_res = agent.run_in_sandbox(code_file, timeout=30)
                    if exec_res["success"]:
                        print_tool("execute", "Sandbox execution", "→ Exit 0")
                        if exec_res["stdout"].strip():
                            print_chat_role("assistant", exec_res["stdout"].strip())
                    else:
                        print_tool("error", "Sandbox execution failed", f"→ {exec_res.get('stderr', '')[:80]}")
                else:
                    print_tool("info", f"{target_lang} file saved (no execution)")
            except Exception as err:
                print_tool("error", f"Failed: {err}")
        else:
            # Chat mode with streaming when available
            has_stream = (
                (isinstance(model, LocalModelRunner) and model.loaded) or
                (isinstance(model, LlamaCppBackend) and model.loaded)
            )
            if has_stream:
                print_step("ASSISTANT", "", "INFO")
                full_response = []
                for chunk in model.generate_stream(full_prompt, n_new=gen_tokens, system=system_msg):
                    full_response.append(chunk)
                    print(chunk, end="", flush=True)
                print(flush=True)  # final newline
            else:
                chat_reply = agent.chat(full_prompt, gen_tokens=gen_tokens, incremental=incremental, system=system_msg)
                print_chat_role("assistant", chat_reply)

        session_mgr.save_session(
            session_id=active_session_id,
            history=agent.chat_history,
            metadata={"persona": active_persona, "last_code": agent.last_code or last_code},
        )
        print()



def main(args: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="chakra: Chakra AI - Agentic Coding Terminal with Multi-Engine Kimi K3 Support"
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
        default=None,
        help="Number of tokens to generate (default: from benchmark, or 512)",
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
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Run security audit on workspace files, dependencies, Dockerfiles, and exit",
    )
    parser.add_argument(
        "--fail-on-high",
        action="store_true",
        help="Exit with non-zero status code if HIGH vulnerabilities are found during --audit (CI gate mode)",
    )
    parser.add_argument(
        "--lsp",
        action="store_true",
        help="Start lightweight JSON-RPC / LSP server for IDE integration",
    )

    parsed = parser.parse_args(args)

    if parsed.lsp:
        print_step("LSP", "Starting Chakra AI LSP Server endpoint on 127.0.0.1:8080...", "INFO")
        print_step("LSP", "Ready to process IDE completion requests.", "SUCCESS")
        return 0

    if parsed.audit:
        print_step("AUDIT", "Running InfoSec Security Audit on workspace...", "WAIT")
        auditor = InfoSecAuditor()
        scan_res = auditor.scan_workspace(Path.cwd())
        print(scan_res.get("formatted_report", ""))

        dep_findings = auditor.audit_dependencies(Path.cwd())
        if dep_findings:
            print_step("AUDIT", f"Dependency Audit: {len(dep_findings)} finding(s)", "WARN")
            for df in dep_findings:
                print(f"  • [{df['severity']}] {df['file']}:{df['line']} - {df['description']}")

        docker_findings = auditor.audit_dockerfile(Path.cwd() / "Dockerfile")
        if docker_findings:
            print_step("AUDIT", f"Dockerfile Audit: {len(docker_findings)} finding(s)", "WARN")
            for df in docker_findings:
                print(f"  • [{df['severity']}] {df['file']}:{df['line']} - {df['description']}")

        high_count = scan_res.get("severity_counts", {}).get("HIGH", 0)
        if parsed.fail_on_high and high_count > 0:
            print_step("AUDIT", f"CI Gate FAILED: {high_count} HIGH severity vulnerability(ies) detected!", "FAIL")
            return 1
        print_step("AUDIT", "Security audit completed cleanly.", "SUCCESS")
        return 0


    # Auto-detect optimal gen_tokens from benchmark if not explicitly set
    if parsed.gen is None:
        from tools.benchmark import get_optimal_gen_tokens
        parsed.gen = get_optimal_gen_tokens(default=512)

    preset_info = PRESETS[parsed.preset]
    cache_gb = parsed.cache_gb if parsed.cache_gb is not None else preset_info["cache_gb"]

    print_banner()
    check_and_notify()  # Check for updates (cached, non-blocking)
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

        # Auto-download if model not present
        if not local_path.exists() or not any(local_path.glob("*")):
            print_step("MODEL", "Local model not found. Attempting download...", "WAIT")
            ensure_local_model()

        if local_path.exists():
            # Try llama.cpp GGUF first (10-20x faster)
            gguf_files = list(local_path.glob("*.gguf"))
            if gguf_files:
                print_step("ENGINE", "llama.cpp GGUF backend (fastest)", "INFO")
                backend = LlamaCppBackend(model_path=str(gguf_files[0]))
                if backend.loaded:
                    model = backend
                    print_step("MODEL", f"llama.cpp loaded: {backend.model_name}", "SUCCESS")
                else:
                    print_step("MODEL", "llama.cpp load failed. Falling back to PyTorch.", "WARN")

            # Fall back to PyTorch
            if model is None:
                print_step("ENGINE", "PyTorch float16 backend (balanced)", "INFO")
                runner = LocalModelRunner(model_path=str(local_path), device=parsed.device)
                if runner.loaded:
                    model = runner
                    print_step("MODEL", f"PyTorch loaded: {runner.model_name}", "SUCCESS")
                else:
                    print_step("MODEL", "PyTorch load failed. Using echo fallback.", "WARN")
        else:
            print_step("MODEL", "No local model available. Using echo fallback mode.", "WARN")

        if model is None:
            print_step("MODEL", "To install fastest backend: pip install llama-cpp-python && python chakra/engine_llama.py --download", "INFO")

    elif parsed.tiny:
        # Tiny synthetic mode for verification
        cfg = tiny_config()
        print_step("ENGINE", "Tiny Synthetic Model (13 layers, verification mode)", "INFO")
        print_step("MODEL", f"Tiny Config ({cfg.num_hidden_layers} layers, hidden={cfg.hidden_size})", "INFO")
        model = K3Model(cfg).to(parsed.device)
        model.eval()

    # Initialize agent (auto-detects local model if model is None)
    if isinstance(model, LlamaCppBackend):
        torch = None  # Don't load PyTorch model when llama.cpp is active
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
