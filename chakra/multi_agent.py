# chakra/multi_agent.py
"""
Multi-Agent Orchestration System for Kimi K3 PyTorch Runtime on Windows.
Author & Creator: Abhirup Guha (Info Security Solution)

Provides ArchitectAgent, CoderAgent, AuditorAgent, and MultiAgentOrchestrator
for offline, multi-module software design, code generation, security auditing,
and sandbox execution on 8GB RAM Windows systems.
"""
from __future__ import annotations

import ast
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from chakra.agent import KimiAgent
from chakra.security import InfoSecAuditor


class ArchitectAgent:
    """
    ArchitectAgent: High-level system architect responsible for blueprint design
    and modular task breakdown of complex user software requests.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        device: str = "cpu",
    ) -> None:
        """
        Initialize ArchitectAgent.

        Args:
            model: K3Model or callable model instance (optional).
            tokenizer: KimiTokenizer instance (optional).
            device: Execution device ('cpu' or 'cuda').
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.kimi_agent = KimiAgent(model=model, tokenizer=tokenizer, device=device)

    def plan_blueprint(self, prompt: str) -> Dict[str, Any]:
        """
        Decomposes a user software request into a modular architectural blueprint.

        Args:
            prompt: User description of the software system to build.

        Returns:
            Dict containing project_name, architecture_summary, and list of module_spec dicts.
        """
        if not prompt or not prompt.strip():
            prompt = "Modular Python Utility System"

        # Attempt model-driven planning if model is available
        if self.model is not None:
            architect_prompt = (
                f"Design a modular Python architecture for the following project request:\n"
                f"'{prompt}'\n"
                f"Return JSON format with fields: 'project_name', 'architecture_summary', 'modules' (list of module specs)."
            )
            try:
                model_output = self.kimi_agent.chat(architect_prompt)
                # Try parsing JSON from model output
                json_match = re.search(r"\{.*\}", model_output, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    if isinstance(parsed, dict) and "modules" in parsed and isinstance(parsed["modules"], list):
                        return parsed
            except Exception:
                pass

        # Robust, offline fallback blueprint synthesis tailored to prompt keywords
        return self._synthesize_fallback_blueprint(prompt)

    def _synthesize_fallback_blueprint(self, prompt: str) -> Dict[str, Any]:
        """
        Synthesizes a structured multi-module architecture blueprint for offline execution.
        """
        p_lower = prompt.lower()

        # Clean project name from prompt
        slug = re.sub(r"[^\w]+", "_", p_lower).strip("_")
        project_name = slug[:30] if slug else "multi_module_app"

        # Special architecture blueprints based on intent
        if "calc" in p_lower or "calculator" in p_lower:
            modules = [
                {
                    "module_name": "config",
                    "filename": "config.py",
                    "purpose": "Calculator settings, constants, and operation definitions.",
                    "specifications": "Define operation symbols, precision settings, and error messages.",
                    "dependencies": [],
                    "interface": ["PRECISION", "SUPPORTED_OPS"],
                },
                {
                    "module_name": "engine",
                    "filename": "engine.py",
                    "purpose": "Core arithmetic calculation logic.",
                    "specifications": "Implement add, subtract, multiply, divide with validation.",
                    "dependencies": ["config.py"],
                    "interface": ["CalculatorEngine"],
                },
                {
                    "module_name": "main",
                    "filename": "main.py",
                    "purpose": "Application entry point and CLI controller.",
                    "specifications": "Instantiate CalculatorEngine and demonstrate calculations.",
                    "dependencies": ["config.py", "engine.py"],
                    "interface": ["main"],
                },
            ]
            summary = "Three-tier modular calculator architecture: Configuration, Calculation Engine, and CLI Controller."

        elif "scan" in p_lower or "port" in p_lower or "net" in p_lower or "security" in p_lower:
            modules = [
                {
                    "module_name": "config",
                    "filename": "config.py",
                    "purpose": "Network scanner targets, default ports, and timeout settings.",
                    "specifications": "Define TARGET_HOSTS, DEFAULT_PORTS, and SCAN_TIMEOUT.",
                    "dependencies": [],
                    "interface": ["DEFAULT_PORTS", "SCAN_TIMEOUT"],
                },
                {
                    "module_name": "scanner",
                    "filename": "scanner.py",
                    "purpose": "Socket port scanning and connectivity check module.",
                    "specifications": "Implement TCP socket connection verification for target ports.",
                    "dependencies": ["config.py"],
                    "interface": ["PortScanner"],
                },
                {
                    "module_name": "main",
                    "filename": "main.py",
                    "purpose": "Security auditing runner script.",
                    "specifications": "Run PortScanner on target ports and print diagnostic results.",
                    "dependencies": ["config.py", "scanner.py"],
                    "interface": ["main"],
                },
            ]
            summary = "Security port scanner architecture: Config, Socket Scanner engine, and Audit runner main script."

        elif "file" in p_lower or "manage" in p_lower or "dir" in p_lower:
            modules = [
                {
                    "module_name": "config",
                    "filename": "config.py",
                    "purpose": "File manager paths and storage configuration.",
                    "specifications": "Define BASE_DIR, ALLOWED_EXTENSIONS, and MAX_FILE_SIZE.",
                    "dependencies": [],
                    "interface": ["BASE_DIR", "ALLOWED_EXTENSIONS"],
                },
                {
                    "module_name": "storage",
                    "filename": "storage.py",
                    "purpose": "File I/O operations, directory creation, and list utilities.",
                    "specifications": "Implement safe file read/write, listing, and directory creation.",
                    "dependencies": ["config.py"],
                    "interface": ["StorageManager"],
                },
                {
                    "module_name": "main",
                    "filename": "main.py",
                    "purpose": "File management CLI interface and orchestration entry point.",
                    "specifications": "Instantiate StorageManager and perform file system operations.",
                    "dependencies": ["config.py", "storage.py"],
                    "interface": ["main"],
                },
            ]
            summary = "File management system architecture: Config settings, StorageManager, and Main CLI interface."

        elif "scrape" in p_lower or "crawl" in p_lower or "web" in p_lower and "scrap" in p_lower:
            modules = [
                {"module_name": "config", "filename": "config.py",
                 "purpose": "Web scraper URLs, headers, and selectors.",
                 "specifications": "Define TARGET_URLS, HEADERS, REQUEST_TIMEOUT, and CSS_SELECTORS.",
                 "dependencies": [], "interface": ["TARGET_URLS", "HEADERS"]},
                {"module_name": "scraper", "filename": "scraper.py",
                 "purpose": "HTTP requests and HTML parsing engine.",
                 "specifications": "Implement fetch_page() and extract_data() using urllib and html.parser.",
                 "dependencies": ["config.py"], "interface": ["WebScraper"]},
                {"module_name": "main", "filename": "main.py",
                 "purpose": "Scraper runner and output formatter.",
                 "specifications": "Run WebScraper on configured URLs and display extracted data.",
                 "dependencies": ["config.py", "scraper.py"], "interface": ["main"]},
            ]
            summary = "Web scraper architecture: Config, Scraper engine (stdlib), and Runner."

        elif "api" in p_lower or "flask" in p_lower or "rest" in p_lower or "endpoint" in p_lower:
            modules = [
                {"module_name": "config", "filename": "config.py",
                 "purpose": "API configuration, routes, and port settings.",
                 "specifications": "Define API_HOST, API_PORT, and ROUTES dict.",
                 "dependencies": [], "interface": ["API_HOST", "API_PORT", "ROUTES"]},
                {"module_name": "server", "filename": "server.py",
                 "purpose": "HTTP server with request routing using http.server stdlib.",
                 "specifications": "Implement RequestHandler with GET/POST route dispatch.",
                 "dependencies": ["config.py"], "interface": ["APIServer"]},
                {"module_name": "main", "filename": "main.py",
                 "purpose": "API server launcher.",
                 "specifications": "Start APIServer on configured host:port.",
                 "dependencies": ["config.py", "server.py"], "interface": ["main"]},
            ]
            summary = "REST API architecture: Config, HTTP Server (stdlib), and Launcher."

        elif "csv" in p_lower or "data" in p_lower or "analyz" in p_lower or "pandas" in p_lower:
            modules = [
                {"module_name": "config", "filename": "config.py",
                 "purpose": "Data analysis paths and column definitions.",
                 "specifications": "Define DATA_FILE, OUTPUT_FILE, and COLUMN_MAPPINGS.",
                 "dependencies": [], "interface": ["DATA_FILE", "OUTPUT_FILE"]},
                {"module_name": "analyzer", "filename": "analyzer.py",
                 "purpose": "CSV parsing and statistical analysis using csv module.",
                 "specifications": "Implement load_csv(), compute_stats(), and generate_report().",
                 "dependencies": ["config.py"], "interface": ["DataAnalyzer"]},
                {"module_name": "main", "filename": "main.py",
                 "purpose": "Data analysis runner and report printer.",
                 "specifications": "Load data, compute statistics, and print formatted report.",
                 "dependencies": ["config.py", "analyzer.py"], "interface": ["main"]},
            ]
            summary = "Data analysis architecture: Config, CSV Analyzer (stdlib), and Report Runner."

        elif "cli" in p_lower or "argparse" in p_lower or "command line" in p_lower:
            modules = [
                {"module_name": "config", "filename": "config.py",
                 "purpose": "CLI application metadata and defaults.",
                 "specifications": "Define APP_NAME, VERSION, and DEFAULT_ARGS.",
                 "dependencies": [], "interface": ["APP_NAME", "VERSION"]},
                {"module_name": "cli", "filename": "cli.py",
                 "purpose": "Argument parser and command dispatcher using argparse.",
                 "specifications": "Build argparse parser with subcommands and dispatch to handlers.",
                 "dependencies": ["config.py"], "interface": ["build_parser", "run_cli"]},
                {"module_name": "main", "filename": "main.py",
                 "purpose": "CLI entry point.",
                 "specifications": "Parse arguments and dispatch to appropriate handler.",
                 "dependencies": ["config.py", "cli.py"], "interface": ["main"]},
            ]
            summary = "CLI tool architecture: Config, Argparse dispatcher, and Entry point."

        elif "hash" in p_lower or "password" in p_lower or "crypt" in p_lower or "encrypt" in p_lower:
            modules = [
                {"module_name": "config", "filename": "config.py",
                 "purpose": "Hashing algorithm settings and salt configuration.",
                 "specifications": "Define HASH_ALGORITHM, SALT_LENGTH, and ITERATIONS.",
                 "dependencies": [], "interface": ["HASH_ALGORITHM", "SALT_LENGTH"]},
                {"module_name": "hasher", "filename": "hasher.py",
                 "purpose": "Secure hashing using hashlib with salt generation.",
                 "specifications": "Implement generate_salt(), hash_password(), and verify_password().",
                 "dependencies": ["config.py"], "interface": ["PasswordHasher"]},
                {"module_name": "main", "filename": "main.py",
                 "purpose": "Hash generator demo and verification runner.",
                 "specifications": "Demo password hashing and verification with user input.",
                 "dependencies": ["config.py", "hasher.py"], "interface": ["main"]},
            ]
            summary = "Password hashing architecture: Config, Hasher (hashlib), and Demo runner."

        elif "todo" in p_lower or "task" in p_lower or "list" in p_lower and "manage" in p_lower:
            modules = [
                {"module_name": "config", "filename": "config.py",
                 "purpose": "Task manager storage path and settings.",
                 "specifications": "Define STORAGE_FILE, MAX_TASKS, and PRIORITY_LEVELS.",
                 "dependencies": [], "interface": ["STORAGE_FILE", "PRIORITY_LEVELS"]},
                {"module_name": "manager", "filename": "manager.py",
                 "purpose": "Task CRUD operations with JSON persistence.",
                 "specifications": "Implement add_task(), complete_task(), list_tasks(), save/load from JSON.",
                 "dependencies": ["config.py"], "interface": ["TaskManager"]},
                {"module_name": "main", "filename": "main.py",
                 "purpose": "Interactive task manager CLI.",
                 "specifications": "Run interactive loop for task management commands.",
                 "dependencies": ["config.py", "manager.py"], "interface": ["main"]},
            ]
            summary = "Task manager architecture: Config, CRUD Manager (JSON), and Interactive CLI."

        else:
            # Generic multi-module blueprint
            modules = [
                {
                    "module_name": "config",
                    "filename": "config.py",
                    "purpose": "System configuration and constants.",
                    "specifications": f"Define configuration settings for task: {prompt}",
                    "dependencies": [],
                    "interface": ["APP_NAME", "VERSION", "load_config"],
                },
                {
                    "module_name": "core",
                    "filename": "core.py",
                    "purpose": "Core business logic processing engine.",
                    "specifications": f"Implement main task processing algorithms for: {prompt}",
                    "dependencies": ["config.py"],
                    "interface": ["CoreEngine", "process_data"],
                },
                {
                    "module_name": "main",
                    "filename": "main.py",
                    "purpose": "Application entrypoint and test execution runner.",
                    "specifications": "Load configuration, instantiate CoreEngine, execute task, and display output.",
                    "dependencies": ["config.py", "core.py"],
                    "interface": ["main"],
                },
            ]
            summary = f"Modular 3-component architecture (Config, Core, Main) designed for: {prompt}"

        return {
            "project_name": project_name,
            "architecture_summary": summary,
            "modules": modules,
        }


class CoderAgent:
    """
    CoderAgent: Component developer agent responsible for generating Python module code
    conforming strictly to specified module specifications and interfaces.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        device: str = "cpu",
        kimi_agent: Optional[KimiAgent] = None,
        agent: Optional[Any] = None,
    ) -> None:
        """
        Initialize CoderAgent.

        Args:
            model: K3Model or callable model instance (optional).
            tokenizer: KimiTokenizer instance (optional).
            device: Execution device ('cpu' or 'cuda').
            kimi_agent: Existing KimiAgent instance (optional).
            agent: Alias for kimi_agent.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.kimi_agent = kimi_agent or agent or KimiAgent(model=model, tokenizer=tokenizer, device=device)

    def generate_module(
        self,
        module_spec: Dict[str, Any],
        context_modules: Optional[Dict[str, str]] = None,
        feedback: Optional[str] = None,
    ) -> str:
        """
        Generates functional Python source code for a given module specification.

        Args:
            module_spec: Dict containing module_name, filename, purpose, specifications, interface, dependencies.
            context_modules: Dict of previously generated filename -> code content.
            feedback: Optional error message or audit feedback for code correction/retry.

        Returns:
            Python source code string.
        """
        filename = module_spec.get("filename", "module.py")
        purpose = module_spec.get("purpose", "Module implementation")
        specifications = module_spec.get("specifications", "")
        dependencies = module_spec.get("dependencies", [])

        # Build context string of previously generated modules if available
        context_str = ""
        if context_modules:
            context_items = []
            for dep in dependencies:
                if dep in context_modules:
                    context_items.append(f"# Dependency {dep}:\n{context_modules[dep]}")
            if context_items:
                context_str = "\n" + "\n".join(context_items) + "\n"

        prompt = (
            f"Write complete, runnable Python module source code for '{filename}'.\n"
            f"Purpose: {purpose}\n"
            f"Specifications: {specifications}\n"
            f"Dependencies: {', '.join(dependencies) if dependencies else 'None'}\n"
            f"{context_str}"
        )
        if feedback:
            prompt += f"\nPrevious attempt had issues/errors: {feedback}\nPlease fix these issues in the code."

        # If model is present and capable, try model-driven generation
        if self.model is not None and callable(self.model):
            try:
                raw_response = self.kimi_agent.chat(prompt)
                extracted_code = self.kimi_agent.extract_code(raw_response)
                # Verify syntax
                ast.parse(extracted_code)
                return extracted_code
            except Exception:
                pass

        # Offline synthesized code generator tailored to module_spec
        return self._synthesize_module_code(module_spec, context_modules, feedback)

    def _synthesize_module_code(
        self,
        module_spec: Dict[str, Any],
        context_modules: Optional[Dict[str, str]] = None,
        feedback: Optional[str] = None,
    ) -> str:
        """
        Synthesizes robust, syntax-valid Python code for offline/low-memory execution environments.
        """
        filename = module_spec.get("filename", "module.py")
        mod_name = module_spec.get("module_name", "module")
        purpose = module_spec.get("purpose", "")
        specs = module_spec.get("specifications", "")

        # 1. Config module
        if mod_name == "config" or filename == "config.py":
            return (
                "# config.py - Application Configuration & Settings\n"
                "\"\"\"\n"
                "Configuration settings and global constants.\n"
                "\"\"\"\n"
                "import os\n"
                "import sys\n\n"
                "APP_NAME = 'KimiPy Multi-Agent Workspace'\n"
                "VERSION = '1.0.0'\n"
                "PRECISION = 4\n"
                "DEFAULT_PORTS = [80, 443, 8080, 22]\n"
                "SCAN_TIMEOUT = 1.0\n"
                "BASE_DIR = os.path.dirname(os.path.abspath(__file__))\n\n"
                "def load_config() -> dict:\n"
                "    \"\"\"Loads configuration properties dictionary.\"\"\"\n"
                "    return {\n"
                "        'app_name': APP_NAME,\n"
                "        'version': VERSION,\n"
                "        'precision': PRECISION,\n"
                "        'ports': DEFAULT_PORTS,\n"
                "        'base_dir': BASE_DIR,\n"
                "    }\n\n"
                "if __name__ == '__main__':\n"
                "    print('Config loaded:', load_config())\n"
            )

        # 2. Calculator Engine module
        if ("calc" in mod_name or "engine" in mod_name) and ("calc" in specs.lower() or "arithmetic" in specs.lower() or "add" in specs.lower()):
            return (
                "# engine.py - Core Calculation Engine\n"
                "\"\"\"\n"
                "Calculation engine module.\n"
                "\"\"\"\n"
                "import sys\n"
                "try:\n"
                "    import config\n"
                "except ImportError:\n"
                "    config = None\n\n"
                "class CalculatorEngine:\n"
                "    \"\"\"Executes arithmetic operations with error handling.\"\"\"\n\n"
                "    def add(self, a: float, b: float) -> float:\n"
                "        return a + b\n\n"
                "    def subtract(self, a: float, b: float) -> float:\n"
                "        return a - b\n\n"
                "    def multiply(self, a: float, b: float) -> float:\n"
                "        return a * b\n\n"
                "    def divide(self, a: float, b: float) -> float:\n"
                "        if b == 0:\n"
                "            raise ValueError('Division by zero is undefined.')\n"
                "        return a / b\n\n"
                "if __name__ == '__main__':\n"
                "    engine = CalculatorEngine()\n"
                "    print('10 + 5 =', engine.add(10, 5))\n"
                "    print('20 / 4 =', engine.divide(20, 4))\n"
            )

        # 3. Port Scanner module
        if "scan" in mod_name or "scanner" in mod_name or "socket" in specs.lower() or "port" in specs.lower():
            return (
                "# scanner.py - Socket Port Scanner Engine\n"
                "\"\"\"\n"
                "Network port auditing module.\n"
                "\"\"\"\n"
                "import socket\n"
                "from typing import List, Dict\n"
                "try:\n"
                "    import config\n"
                "except ImportError:\n"
                "    config = None\n\n"
                "class PortScanner:\n"
                "    \"\"\"Checks TCP port connectivity on target hosts.\"\"\"\n\n"
                "    def __init__(self, timeout: float = 0.5) -> None:\n"
                "        self.timeout = timeout if config is None else getattr(config, 'SCAN_TIMEOUT', timeout)\n\n"
                "    def check_port(self, host: str, port: int) -> bool:\n"
                "        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
                "        s.settimeout(self.timeout)\n"
                "        try:\n"
                "            res = s.connect_ex((host, port))\n"
                "            return res == 0\n"
                "        except Exception:\n"
                "            return False\n"
                "        finally:\n"
                "            s.close()\n\n"
                "    def scan_ports(self, host: str, ports: List[int]) -> Dict[int, bool]:\n"
                "        results = {}\n"
                "        for p in ports:\n"
                "            results[p] = self.check_port(host, p)\n"
                "        return results\n\n"
                "if __name__ == '__main__':\n"
                "    scanner = PortScanner()\n"
                "    ports = getattr(config, 'DEFAULT_PORTS', [80, 443]) if config else [80, 443]\n"
                "    res = scanner.scan_ports('127.0.0.1', ports)\n"
                "    print('Scan Results:', res)\n"
            )

        # 4. Storage Manager module
        if "storage" in mod_name or "file" in mod_name or "dir" in specs.lower() or "io" in specs.lower():
            return (
                "# storage.py - Safe File Storage Manager\n"
                "\"\"\"\n"
                "File management and storage operation utilities.\n"
                "\"\"\"\n"
                "import os\n"
                "from pathlib import Path\n"
                "from typing import List\n"
                "try:\n"
                "    import config\n"
                "except ImportError:\n"
                "    config = None\n\n"
                "class StorageManager:\n"
                "    \"\"\"Manages local workspace file read/write operations safely.\"\"\"\n\n"
                "    def write_file(self, filename: str, content: str) -> str:\n"
                "        path = Path(filename)\n"
                "        path.parent.mkdir(parents=True, exist_ok=True)\n"
                "        path.write_text(content, encoding='utf-8')\n"
                "        return str(path.resolve())\n\n"
                "    def read_file(self, filename: str) -> str:\n"
                "        path = Path(filename)\n"
                "        if not path.exists():\n"
                "            raise FileNotFoundError(f'File not found: {filename}')\n"
                "        return path.read_text(encoding='utf-8')\n\n"
                "    def list_files(self, dir_path: str = '.') -> List[str]:\n"
                "        path = Path(dir_path)\n"
                "        if not path.exists():\n"
                "            return []\n"
                "        return [f.name for f in path.iterdir() if f.is_file()]\n\n"
                "if __name__ == '__main__':\n"
                "    mgr = StorageManager()\n"
                "    print('Current files:', mgr.list_files('.'))\n"
            )

        # 5. Generic Core Business Logic module
        if mod_name == "core" or filename == "core.py":
            return (
                "# core.py - Core Task Engine\n"
                "\"\"\"\n"
                "Core business logic processing engine.\n"
                "\"\"\"\n"
                "import os\n"
                "import sys\n"
                "try:\n"
                "    import config\n"
                "except ImportError:\n"
                "    config = None\n\n"
                "class CoreEngine:\n"
                "    \"\"\"Core business logic engine.\"\"\"\n\n"
                "    def __init__(self) -> None:\n"
                "        self.config = config.load_config() if (config and hasattr(config, 'load_config')) else {}\n\n"
                "    def process_data(self, data: str) -> dict:\n"
                "        \"\"\"Processes input data payload and returns structured status.\"\"\"\n"
                "        cleaned = data.strip()\n"
                "        return {\n"
                "            'status': 'SUCCESS',\n"
                "            'processed_length': len(cleaned),\n"
                "            'summary': f'Processed input of {len(cleaned)} characters.',\n"
                "        }\n\n"
                "if __name__ == '__main__':\n"
                "    engine = CoreEngine()\n"
                "    print('Core Engine Data:', engine.process_data('Multi-Agent Test Data'))\n"
            )

        # 6. Main Entry Point module
        if mod_name == "main" or filename == "main.py":
            # Inspect context_modules to build imports dynamically
            imports_str = ""
            has_engine = context_modules and "engine.py" in context_modules
            has_scanner = context_modules and "scanner.py" in context_modules
            has_storage = context_modules and "storage.py" in context_modules
            has_core = context_modules and "core.py" in context_modules

            body_lines = ["print('=== Multi-Agent System Execution ===')"]
            body_lines.append("try:\n        import config\n        print('Loaded app:', getattr(config, 'APP_NAME', 'KimiPy'))\n    except Exception as e:\n        print('Config note:', e)")

            if has_engine:
                imports_str += "from engine import CalculatorEngine\n"
                body_lines.append("calc = CalculatorEngine()\n    print('Calculator Test (15 + 25):', calc.add(15, 25))")

            if has_scanner:
                imports_str += "from scanner import PortScanner\n"
                body_lines.append("scanner = PortScanner()\n    res = scanner.scan_ports('127.0.0.1', [80, 443])\n    print('Port Scan Test:', res)")

            if has_storage:
                imports_str += "from storage import StorageManager\n"
                body_lines.append("storage = StorageManager()\n    print('Storage Files Test:', storage.list_files('.'))")

            if has_core or (not has_engine and not has_scanner and not has_storage):
                imports_str += "try:\n    from core import CoreEngine\nexcept ImportError:\n    CoreEngine = None\n"
                body_lines.append("if CoreEngine:\n        core = CoreEngine()\n        print('Core Engine Test:', core.process_data('Integration System Test'))\n    else:\n        print('Execution verified successfully.')")

            indented_body = "\n    ".join(body_lines)

            return (
                "# main.py - Application Entry Point & System Runner\n"
                "\"\"\"\n"
                "Main entry point for multi-agent system execution.\n"
                "\"\"\"\n"
                "import sys\n"
                "import os\n"
                f"{imports_str}\n"
                "def main() -> int:\n"
                "    \"\"\"Main orchestration entry point.\"\"\"\n"
                f"    {indented_body}\n"
                "    return 0\n\n"
                "if __name__ == '__main__':\n"
                "    sys.exit(main())\n"
            )

        # Default fallback module
        return (
            f"# {filename} - {purpose}\n"
            "\"\"\"\n"
            f"{specs}\n"
            "\"\"\"\n"
            "import os\n"
            "import sys\n\n"
            "def run_module() -> bool:\n"
            f"    print('Executing module: {mod_name}')\n"
            "    return True\n\n"
            "if __name__ == '__main__':\n"
            "    run_module()\n"
        )


class AuditorAgent:
    """
    AuditorAgent: Windows sandbox testing and security/error auditing agent.
    Delegates static AST security analysis to InfoSecAuditor for comprehensive OWASP coverage.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.kimi_agent = KimiAgent(device=device)
        self.infosec_auditor = InfoSecAuditor()

    def audit_security(self, code: str) -> Dict[str, Any]:
        """
        Performs static analysis via InfoSecAuditor for comprehensive security coverage.

        Returns:
            Dict containing score (0-100), issues list, and has_critical flag.
        """
        if not code or not code.strip():
            return {
                "score": 0,
                "issues": [{"level": "CRITICAL", "message": "Code snippet is empty.", "line": 0}],
                "has_critical": True,
            }

        result = self.infosec_auditor.audit_code(code)
        issues = []
        has_critical = False
        for v in result.get("vulnerabilities", []):
            severity = v.get("severity", "MEDIUM")
            level = "CRITICAL" if severity == "HIGH" else ("WARNING" if severity == "MEDIUM" else "INFO")
            if severity == "HIGH":
                has_critical = True
            issues.append({
                "level": level,
                "message": v.get("description", v.get("type", "Security issue")),
                "line": v.get("line", 0),
            })

        return {
            "score": result.get("score", 0),
            "issues": issues,
            "has_critical": has_critical or result.get("syntax_error") is not None,
        }

    def audit_and_test(
        self,
        code: str,
        filename: str = "module.py",
        cwd: Optional[str] = None,
        timeout: int = 10,
    ) -> Dict[str, Any]:
        """
        Performs full security audit and Windows sandbox runtime testing.

        Args:
            code: Source code string to audit and execute.
            filename: Name of the script/module file.
            cwd: Optional working directory for sandbox execution.
            timeout: Subprocess timeout in seconds.

        Returns:
            Dict containing passed flag, security_score, security_issues, and execution_result.
        """
        # Step 1: Static security audit
        sec_report = self.audit_security(code)

        # Step 2: Isolated sandbox execution
        exec_res = self.kimi_agent.execute_sandbox(code, timeout=timeout, cwd=cwd)

        passed = exec_res["success"] and not sec_report["has_critical"]

        return {
            "passed": passed,
            "security_score": sec_report["score"],
            "security_issues": sec_report["issues"],
            "execution_result": exec_res,
            "stdout": exec_res["stdout"],
            "stderr": exec_res["stderr"],
            "exit_code": exec_res["returncode"],
        }


class MultiAgentOrchestrator:
    """
    MultiAgentOrchestrator: Lead orchestrator class coordinating ArchitectAgent,
    CoderAgent, and AuditorAgent to build complete multi-module projects offline.
    """

    def __init__(
        self,
        architect: Optional[ArchitectAgent] = None,
        coder: Optional[CoderAgent] = None,
        auditor: Optional[AuditorAgent] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        device: str = "cpu",
        agent: Optional[Any] = None,
    ) -> None:
        """
        Initialize MultiAgentOrchestrator.

        Args:
            architect: ArchitectAgent instance (optional).
            coder: CoderAgent instance (optional).
            auditor: AuditorAgent instance (optional).
            model: K3Model or callable model instance (optional).
            tokenizer: KimiTokenizer instance (optional).
            device: Execution device ('cpu' or 'cuda').
            agent: KimiAgent instance (optional).
        """
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.agent = agent
        self.architect = architect or ArchitectAgent(model=model, tokenizer=tokenizer, device=device)
        self.coder = coder or CoderAgent(model=model, tokenizer=tokenizer, device=device, agent=agent)
        self.auditor = auditor or AuditorAgent(device=device)

    def list_agents(self) -> Dict[str, str]:
        """Returns active team sub-agent roles and descriptions."""
        return {
            "Supervisor": "Coordinates team workflow, delegates tasks, and approves final execution.",
            "Architect": "Analyzes requirements and designs structured implementation plans.",
            "Coder": "Generates, synthesizes, and refines high-quality Python code implementations.",
            "Auditor": "Audits generated code for syntax, logic, security, and verifies sandbox execution.",
        }

    def run_team_collaboration(
        self,
        prompt: str,
        max_retries: int = 3,
        gen_tokens: int = 128,
        incremental: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes Multi-Agent team collaboration mode displaying status badges for Supervisor, Architect, Coder, and Auditor.

        Args:
            prompt: Task prompt string.
            max_retries: Maximum fix attempts.
            gen_tokens: Generation token limit.
            incremental: Incremental model flag.

        Returns:
            Dict containing execution success, generated code, stdout, stderr, and iterations.
        """
        from chakra.ui import print_agent_step

        print_agent_step("Supervisor", f"Initiating multi-agent collaboration for prompt: '{prompt}'")

        # Architect Stage
        print_agent_step("Architect", "Analyzing task requirements and generating architectural blueprint...")
        blueprint = self.architect.plan_blueprint(prompt)
        print_agent_step("Architect", f"Architecture plan created ({len(blueprint.get('modules', []))} modules planned).")

        # Coder Stage
        print_agent_step("Coder", "Generating Python implementation from architectural plan...")
        team_res = self.run_team_task(prompt=prompt, max_retries=max_retries)
        modules = team_res.get("modules", {})
        main_code = ""
        if "main.py" in modules:
            main_code = modules["main.py"]
        elif modules:
            main_code = list(modules.values())[0]
        print_agent_step("Coder", "Code implementation completed. Forwarding to Auditor.")

        # Auditor Stage
        print_agent_step("Auditor", "Auditing generated modules for syntax, security, and sandbox execution...")
        if team_res.get("success", False):
            print_agent_step("Auditor", "Audit PASSED: All modules verified cleanly in sandbox execution.")
            print_agent_step("Supervisor", "Team collaboration complete. Solution verified and approved.")
        else:
            print_agent_step("Auditor", "Audit WARN: Sandbox execution returned warnings or non-zero exit.")
            print_agent_step("Supervisor", "Team collaboration concluded with audit warnings.")

        return {
            "success": team_res.get("success", False),
            "code": main_code,
            "stdout": team_res.get("integration_result", {}).get("stdout", ""),
            "stderr": team_res.get("integration_result", {}).get("stderr", ""),
            "iterations": len(team_res.get("history", [1])),
            "team_result": team_res,
        }

    def run_team_task(

        self,
        prompt: str,
        output_dir: Optional[Union[str, Path]] = None,
        max_retries: int = 3,
        timeout: int = 10,
    ) -> Dict[str, Any]:
        """
        Executes a multi-agent team task from prompt to verified multi-module application.

        Args:
            prompt: High-level request describing the software system to build.
            output_dir: Destination directory path to save generated project files.
            max_retries: Maximum fix/retry attempts per module if audit/test fails.
            timeout: Execution timeout limit per module test in seconds.

        Returns:
            Dict containing overall success status, blueprint, generated modules code map,
            audit reports, project output directory, and execution history.
        """
        # Step 1: Architectural Planning
        blueprint = self.architect.plan_blueprint(prompt)

        # Setup destination directory
        if output_dir is not None:
            workspace = Path(output_dir)
            workspace.mkdir(parents=True, exist_ok=True)
        else:
            temp_dir_obj = tempfile.TemporaryDirectory()
            workspace = Path(temp_dir_obj.name)

        modules_code: Dict[str, str] = {}
        audit_reports: Dict[str, Dict[str, Any]] = {}
        history: List[Dict[str, Any]] = []

        modules_specs = blueprint.get("modules", [])

        # Step 2: Component Code Generation & Audit Iteration Loop
        for spec in modules_specs:
            filename = spec.get("filename", "module.py")
            feedback: Optional[str] = None
            final_code = ""
            final_audit: Dict[str, Any] = {}

            for attempt in range(1, max_retries + 1):
                # Generate component code
                code = self.coder.generate_module(
                    module_spec=spec,
                    context_modules=modules_code,
                    feedback=feedback,
                )
                final_code = code

                # Save code temporarily to workspace for testing module imports
                temp_file = workspace / filename
                temp_file.write_text(code, encoding="utf-8")

                # Audit and execute test in sandbox with cwd set to workspace
                audit_res = self.auditor.audit_and_test(
                    code=code,
                    filename=filename,
                    cwd=str(workspace),
                    timeout=timeout,
                )
                final_audit = audit_res

                history.append(
                    {
                        "module": filename,
                        "attempt": attempt,
                        "code": code,
                        "audit_report": audit_res,
                    }
                )

                if audit_res["passed"]:
                    break

                # Formulate feedback for retry
                err_msg = audit_res["stderr"] or (
                    "; ".join([iss["message"] for iss in audit_res["security_issues"]])
                    if audit_res["security_issues"]
                    else "Execution failed"
                )
                feedback = err_msg

            # Save final module version into modules_code and audit_reports
            modules_code[filename] = final_code
            audit_reports[filename] = final_audit

        # Step 3: Overall Integration Audit & Test
        main_entry = "main.py" if "main.py" in modules_code else (modules_specs[0]["filename"] if modules_specs else None)
        integration_res: Dict[str, Any] = {"passed": True, "stdout": "", "stderr": ""}

        if main_entry and (workspace / main_entry).exists():
            main_code = (workspace / main_entry).read_text(encoding="utf-8")
            integration_res = self.auditor.audit_and_test(
                code=main_code,
                filename=main_entry,
                cwd=str(workspace),
                timeout=timeout,
            )

        overall_success = all(rep.get("passed", False) for rep in audit_reports.values()) and integration_res.get("passed", False)

        return {
            "success": overall_success,
            "blueprint": blueprint,
            "modules": modules_code,
            "audit_reports": audit_reports,
            "output_dir": str(workspace.resolve()),
            "history": history,
            "integration_result": integration_res,
        }
