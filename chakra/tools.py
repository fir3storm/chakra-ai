"""
Tool Calling Protocol for Chakra AI.
Author & Creator: Abhirup Guha (Info Security Solution)

Grammar-constrained tool calling loop for the llama.cpp (engine C) backend: the model chooses
CALL_TOOL vs FINAL_ANSWER via a 1-token grammar-constrained decision, then (if calling a tool)
emits `{"tool": <enum of registered names>, "args": {...}}` constrained via
`LlamaGrammar.from_json_schema`. This gives deterministic, always-parseable tool calls even from
a small (1.5B) model, instead of free-text ReAct parsing that a small model frequently malforms.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from chakra.agent import KimiAgent, apply_search_replace_block
from chakra.security import InfoSecAuditor
from chakra.workspace import WorkspaceIndexer, scan_ast_symbols

MAX_RESULT_CHARS = 1200
DECISION_CHOICES = ["CALL_TOOL", "FINAL_ANSWER"]
# Rough chars-per-token estimate to keep the running conversation under the model's context
# window without needing a real tokenizer call on every turn.
CHARS_PER_TOKEN_ESTIMATE = 4
DEFAULT_CONTEXT_BUDGET_TOKENS = 3200


def _truncate(text: Any, limit: int = MAX_RESULT_CHARS) -> str:
    text = "" if text is None else str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated, {len(text) - limit} more chars]"


@dataclass
class Tool:
    """A callable tool: name, description, JSON-schema for its `args` object, and handler."""

    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[..., Any]


class WorkspaceGuard:
    """Confines file paths to a workspace root; rejects traversal outside it."""

    def __init__(self, root: Optional[str] = None) -> None:
        self.root = Path(root or Path.cwd()).resolve()

    def resolve(self, rel_path: str) -> Path:
        candidate = (self.root / rel_path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError:
            raise PermissionError(f"Path '{rel_path}' escapes workspace root '{self.root}'")
        return candidate


class ToolRegistry:
    """Registers and executes named tools."""

    def __init__(self) -> None:
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    def names(self) -> List[str]:
        return list(self.tools.keys())

    def subset(self, names: List[str]) -> "ToolRegistry":
        """Returns a new registry containing only the named tools (missing names are skipped) —
        used to scope each multi-agent role to the tools appropriate for it."""
        sub = ToolRegistry()
        for name in names:
            tool = self.tools.get(name)
            if tool is not None:
                sub.register(tool)
        return sub

    def describe(self) -> str:
        """Human/model-readable listing of tools and their argument shapes."""
        lines = []
        for tool in self.tools.values():
            props = tool.parameters.get("properties", {})
            arg_hint = ", ".join(props.keys())
            lines.append(f"- {tool.name}({arg_hint}): {tool.description}")
        return "\n".join(lines)

    def execute(self, name: str, args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        tool = self.tools.get(name)
        if tool is None:
            return {"success": False, "error": f"Unknown tool '{name}'"}
        try:
            result = tool.handler(**(args or {}))
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def execute_parallel(self, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Executes multiple {"tool": name, "args": {...}} calls concurrently."""
        import concurrent.futures

        if not calls:
            return []
        if len(calls) == 1:
            c = calls[0]
            return [self.execute(c.get("tool", ""), c.get("args", {}))]

        results: List[Any] = [None] * len(calls)
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(calls), 8)) as executor:
            future_to_idx = {
                executor.submit(self.execute, c.get("tool", ""), c.get("args", {})): idx
                for idx, c in enumerate(calls)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = {"success": False, "error": str(e)}
        return results


def build_default_tools(
    workspace_root: Optional[str] = None,
    kimi_agent: Optional[KimiAgent] = None,
) -> ToolRegistry:
    """Builds the standard Chakra AI tool set, confined to `workspace_root`."""
    guard = WorkspaceGuard(workspace_root)
    agent = kimi_agent or KimiAgent()
    auditor = InfoSecAuditor()
    registry = ToolRegistry()

    def read_file(path: str) -> str:
        p = guard.resolve(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return _truncate(p.read_text(encoding="utf-8", errors="replace"))

    def write_file(path: str, content: str) -> str:
        p = guard.resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {path}"

    def edit_file(path: str, search: str, replace: str) -> str:
        p = guard.resolve(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        original = p.read_text(encoding="utf-8", errors="replace")
        updated, matched = apply_search_replace_block(original, search, replace)
        if not matched:
            raise ValueError("search block not found in file; no changes made")
        p.write_text(updated, encoding="utf-8")
        return f"Patched {path} ({len(updated)} chars)"

    def list_dir(path: str = ".") -> str:
        indexer = WorkspaceIndexer(root_dir=guard.resolve(path))
        indexer.scan_workspace()
        return _truncate(indexer.get_tree())

    def search_symbols(query: str, path: str = ".") -> str:
        indexer = WorkspaceIndexer(root_dir=guard.resolve(path))
        graph = scan_ast_symbols(indexer)
        matches = {n: locs for n, locs in graph.symbols.items() if query.lower() in n.lower()}
        if not matches:
            return f"No symbols matching '{query}'"
        lines = [
            f"{name} ({loc.get('type')}) in {loc.get('file')}"
            for name, locs in matches.items()
            for loc in locs
        ]
        return _truncate("\n".join(lines))

    def run_python(code: str, timeout: int = 10) -> Dict[str, Any]:
        res = agent.execute_sandbox(code, timeout=timeout, cwd=str(guard.root), language="python")
        return {
            "success": res["success"],
            "stdout": _truncate(res["stdout"]),
            "stderr": _truncate(res["stderr"]),
        }

    def run_tests(cmd: str = "pytest") -> Dict[str, Any]:
        try:
            result = subprocess.run(
                cmd.split(), capture_output=True, text=True, timeout=60, cwd=str(guard.root)
            )
            return {
                "success": result.returncode == 0,
                "stdout": _truncate(result.stdout),
                "stderr": _truncate(result.stderr),
            }
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": str(e)}

    def security_audit(code: str) -> Dict[str, Any]:
        result = auditor.audit_code(code)
        return {
            "score": result.get("score", 0),
            "vulnerabilities": _truncate(json.dumps(result.get("vulnerabilities", []), default=str)),
        }

    def git_status() -> str:
        result = subprocess.run(
            ["git", "status", "--short"], capture_output=True, text=True, timeout=15, cwd=str(guard.root)
        )
        return _truncate(result.stdout or result.stderr)

    def git_diff(path: str = "") -> str:
        cmd = ["git", "diff"] + ([path] if path else [])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=str(guard.root))
        return _truncate(result.stdout or result.stderr)

    registry.register(Tool(
        name="read_file",
        description="Read a text file from the workspace and return its contents.",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        handler=read_file,
    ))
    registry.register(Tool(
        name="write_file",
        description="Create or overwrite a file in the workspace with the given content.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
        handler=write_file,
    ))
    registry.register(Tool(
        name="edit_file",
        description="Patch a file: replace an exact `search` text block with `replace`. Use for "
                     "targeted fixes instead of rewriting a whole file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "search": {"type": "string"},
                "replace": {"type": "string"},
            },
            "required": ["path", "search", "replace"],
        },
        handler=edit_file,
    ))
    registry.register(Tool(
        name="list_dir",
        description="List the workspace file tree starting at `path` (default: workspace root).",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": []},
        handler=list_dir,
    ))
    registry.register(Tool(
        name="search_symbols",
        description="Search workspace Python functions/classes by name substring.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}, "path": {"type": "string"}},
            "required": ["query"],
        },
        handler=search_symbols,
    ))
    registry.register(Tool(
        name="run_python",
        description="Execute a Python code snippet in an isolated sandbox; returns stdout/stderr/success.",
        parameters={
            "type": "object",
            "properties": {"code": {"type": "string"}, "timeout": {"type": "integer"}},
            "required": ["code"],
        },
        handler=run_python,
    ))
    registry.register(Tool(
        name="run_tests",
        description="Run a test command (default 'pytest') in the workspace; returns stdout/stderr/success.",
        parameters={"type": "object", "properties": {"cmd": {"type": "string"}}, "required": []},
        handler=run_tests,
    ))
    registry.register(Tool(
        name="security_audit",
        description="Run an OWASP-style static security audit on a code snippet.",
        parameters={"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
        handler=security_audit,
    ))
    registry.register(Tool(
        name="git_status",
        description="Show `git status --short` for the workspace.",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=git_status,
    ))
    registry.register(Tool(
        name="git_diff",
        description="Show `git diff` for the workspace, optionally scoped to one file path.",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": []},
        handler=git_diff,
    ))

    return registry


class ToolLoop:
    """Bounded tool-calling loop, driven by a real growing multi-turn conversation (not a
    flattened re-sent string) so the backend can reuse the shared prefix across turns.

    Each turn: the model picks CALL_TOOL or FINAL_ANSWER (grammar-constrained to those two
    literals). On CALL_TOOL it emits `{"tool": <enum of registered names>, "args": {...}}`
    (grammar-constrained via the backend's `generate_json`), the tool runs, and an assistant
    turn describing the call plus a user turn with its (truncated) result are appended to
    `self.messages` for the next turn. Bounded by `max_steps` and by an approximate context
    token budget (forces FINAL_ANSWER once the conversation is getting close to the model's
    context window, rather than overflowing it).

    Falls back to a single plain-text generation (no tool use) when the backend doesn't expose
    `generate_choice`/`generate_json` (e.g. no model loaded, or a non-grammar-capable backend).
    """

    def __init__(
        self,
        backend: Any,
        registry: ToolRegistry,
        system_prompt: str = "",
        max_steps: int = 6,
        context_budget_tokens: int = DEFAULT_CONTEXT_BUDGET_TOKENS,
    ) -> None:
        self.backend = backend
        self.registry = registry
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.context_budget_tokens = context_budget_tokens

    def _supports_grammar(self) -> bool:
        return hasattr(self.backend, "generate_choice") and hasattr(self.backend, "generate_json")

    def _plain_generate(self, messages: List[Dict[str, str]]) -> str:
        if self.backend is not None and hasattr(self.backend, "generate"):
            return self.backend.generate(messages=messages)
        return ""

    def _approx_tokens(self, messages: List[Dict[str, str]]) -> int:
        total_chars = sum(len(m.get("content", "")) for m in messages)
        return total_chars // CHARS_PER_TOKEN_ESTIMATE

    def run(
        self,
        task_prompt: str,
        forced_first_call: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Runs the loop for a task prompt.

        Args:
            task_prompt: The task/instruction for the model.
            forced_first_call: Optional `{"tool": name, "args": {...}}` to execute as step 1
                without asking the model to decide. Use this when a tool call is deterministically
                known to be needed first (e.g. "read the file before patching it") — a small
                model's CALL_TOOL/FINAL_ANSWER judgment is a blind single-token guess with no
                reasoning behind it, and is unreliable exactly on steps like this; skipping the
                guess for known-necessary steps and reserving the model's judgment for genuinely
                ambiguous decisions measurably improves reliability.

        Returns:
            Dict with `success`, `final_answer`, `tool_calls` (transcript of
            `{tool, args, result}`), and `steps` used.
        """
        transcript: List[Dict[str, Any]] = []

        if not self._supports_grammar():
            messages = [{"role": "user", "content": task_prompt}]
            if self.system_prompt:
                messages.insert(0, {"role": "system", "content": self.system_prompt})
            answer = self._plain_generate(messages)
            return {"success": True, "final_answer": answer, "tool_calls": transcript, "steps": 0}

        tool_names = self.registry.names()
        seen_calls = set()

        system_content = self.system_prompt or ""
        if tool_names:
            system_content = (
                f"{system_content}\n\nAvailable tools:\n{self.registry.describe()}\n"
                f"Call a tool to gather information or take action, or give your final answer."
            ).strip()

        messages: List[Dict[str, str]] = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": task_prompt})

        start_step = 1
        if forced_first_call and forced_first_call.get("tool") in tool_names:
            tool_name = forced_first_call["tool"]
            tool_args = forced_first_call.get("args") or {}
            result = self.registry.execute(tool_name, tool_args)
            transcript.append({"tool": tool_name, "args": tool_args, "result": result})
            seen_calls.add((tool_name, json.dumps(tool_args, sort_keys=True, default=str)))
            messages.append({
                "role": "assistant",
                "content": f"Calling tool {tool_name}({json.dumps(tool_args, default=str)})",
            })
            messages.append({
                "role": "user",
                "content": f"Tool result: {_truncate(json.dumps(result, default=str))}\n\n"
                            f"Continue: call another tool if needed, or give your final answer.",
            })
            start_step = 2

        for step in range(start_step, self.max_steps + 1):
            over_budget = self._approx_tokens(messages) >= self.context_budget_tokens

            decision = (
                "FINAL_ANSWER"
                if over_budget or not tool_names
                else self.backend.generate_choice(choices=DECISION_CHOICES, messages=messages)
            )

            if decision != "CALL_TOOL":
                answer = self._plain_generate(messages)
                return {"success": True, "final_answer": answer, "tool_calls": transcript, "steps": step}

            union_schema = {
                "type": "object",
                "properties": {
                    "tool": {"enum": tool_names},
                    "args": {"type": "object"},
                },
                "required": ["tool", "args"],
            }
            call_messages = messages + [
                {"role": "user", "content": 'Respond with one tool call as JSON: {"tool": <name>, "args": {...}}'}
            ]
            call = self.backend.generate_json(schema=union_schema, messages=call_messages)

            if not call or "tool" not in call:
                # Grammar/parse failure — treat as final answer rather than loop forever.
                answer = self._plain_generate(messages)
                return {"success": True, "final_answer": answer, "tool_calls": transcript, "steps": step}

            tool_name = call.get("tool")
            tool_args = call.get("args") or {}
            call_signature = (tool_name, json.dumps(tool_args, sort_keys=True, default=str))

            if call_signature in seen_calls:
                # Model repeated an identical call — nudge it to answer instead of looping.
                messages.append({
                    "role": "user",
                    "content": f"You already called {tool_name} with those exact args. "
                                f"Use the result above and give your final answer now.",
                })
                continue
            seen_calls.add(call_signature)

            result = self.registry.execute(tool_name, tool_args)
            transcript.append({"tool": tool_name, "args": tool_args, "result": result})

            result_text = _truncate(json.dumps(result, default=str))
            messages.append({
                "role": "assistant",
                "content": f"Calling tool {tool_name}({json.dumps(tool_args, default=str)})",
            })
            messages.append({
                "role": "user",
                "content": f"Tool result: {result_text}\n\nContinue: call another tool if needed, "
                            f"or give your final answer.",
            })

        answer = self._plain_generate(messages)
        return {"success": False, "final_answer": answer, "tool_calls": transcript, "steps": self.max_steps}
