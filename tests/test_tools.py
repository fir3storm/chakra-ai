"""
Unit tests for chakra/tools.py: ToolRegistry, WorkspaceGuard, and ToolLoop.

Uses scripted fake backends (no real model/GGUF needed) that mimic LlamaCppBackend's
generate_choice/generate_json/generate interface, so these tests are fast and deterministic.
"""

from pathlib import Path

import pytest

from chakra.tools import (
    Tool,
    ToolRegistry,
    ToolLoop,
    WorkspaceGuard,
    build_default_tools,
)


# ── ToolRegistry ──────────────────────────────────────────────────────────

def test_tool_registry_register_and_execute():
    reg = ToolRegistry()
    reg.register(Tool(
        name="add",
        description="Add two numbers",
        parameters={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
        handler=lambda a, b: a + b,
    ))
    res = reg.execute("add", {"a": 2, "b": 3})
    assert res["success"] is True
    assert res["result"] == 5


def test_tool_registry_unknown_tool():
    reg = ToolRegistry()
    res = reg.execute("nope", {})
    assert res["success"] is False
    assert "Unknown tool" in res["error"]


def test_tool_registry_handler_exception_is_caught():
    reg = ToolRegistry()
    reg.register(Tool(name="boom", description="", parameters={}, handler=lambda: 1 / 0))
    res = reg.execute("boom", {})
    assert res["success"] is False
    assert "error" in res


def test_tool_registry_subset():
    reg = ToolRegistry()
    reg.register(Tool(name="a", description="", parameters={}, handler=lambda: 1))
    reg.register(Tool(name="b", description="", parameters={}, handler=lambda: 2))
    sub = reg.subset(["a", "nonexistent"])
    assert sub.names() == ["a"]


def test_tool_registry_execute_parallel():
    reg = ToolRegistry()
    reg.register(Tool(name="double", description="", parameters={}, handler=lambda x: x * 2))
    results = reg.execute_parallel([
        {"tool": "double", "args": {"x": 1}},
        {"tool": "double", "args": {"x": 2}},
        {"tool": "double", "args": {"x": 3}},
    ])
    assert [r["result"] for r in results] == [2, 4, 6]


# ── WorkspaceGuard ────────────────────────────────────────────────────────

def test_workspace_guard_resolves_inside_root(tmp_path):
    guard = WorkspaceGuard(str(tmp_path))
    resolved = guard.resolve("sub/file.txt")
    assert str(resolved).startswith(str(tmp_path))


def test_workspace_guard_rejects_traversal(tmp_path):
    guard = WorkspaceGuard(str(tmp_path))
    with pytest.raises(PermissionError):
        guard.resolve("../outside.txt")


def test_workspace_guard_rejects_absolute_escape(tmp_path):
    guard = WorkspaceGuard(str(tmp_path))
    other = tmp_path.parent / "elsewhere.txt"
    with pytest.raises(PermissionError):
        guard.resolve(str(other))


# ── build_default_tools ───────────────────────────────────────────────────

def test_read_write_file_roundtrip(tmp_path):
    reg = build_default_tools(workspace_root=str(tmp_path))
    write_res = reg.execute("write_file", {"path": "a.txt", "content": "hello"})
    assert write_res["success"] is True
    read_res = reg.execute("read_file", {"path": "a.txt"})
    assert read_res["success"] is True
    assert read_res["result"] == "hello"


def test_read_file_missing_raises(tmp_path):
    reg = build_default_tools(workspace_root=str(tmp_path))
    res = reg.execute("read_file", {"path": "missing.txt"})
    assert res["success"] is False


def test_edit_file_applies_patch(tmp_path):
    (tmp_path / "m.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    reg = build_default_tools(workspace_root=str(tmp_path))
    res = reg.execute("edit_file", {"path": "m.py", "search": "return 1", "replace": "return 2"})
    assert res["success"] is True
    assert (tmp_path / "m.py").read_text(encoding="utf-8") == "def f():\n    return 2\n"


def test_edit_file_no_match_fails(tmp_path):
    (tmp_path / "m.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    reg = build_default_tools(workspace_root=str(tmp_path))
    res = reg.execute("edit_file", {"path": "m.py", "search": "nonexistent line", "replace": "x"})
    assert res["success"] is False


def test_run_python_tool_executes(tmp_path):
    reg = build_default_tools(workspace_root=str(tmp_path))
    res = reg.execute("run_python", {"code": "print('hi')"})
    assert res["success"] is True
    assert res["result"]["success"] is True
    assert "hi" in res["result"]["stdout"]


def test_security_audit_tool_flags_eval(tmp_path):
    reg = build_default_tools(workspace_root=str(tmp_path))
    res = reg.execute("security_audit", {"code": "eval('2+2')"})
    assert res["success"] is True
    assert "eval" in res["result"]["vulnerabilities"].lower() or res["result"]["score"] < 100


def test_path_traversal_rejected_through_registry(tmp_path):
    reg = build_default_tools(workspace_root=str(tmp_path))
    res = reg.execute("read_file", {"path": "../outside.txt"})
    assert res["success"] is False
    assert "escapes workspace" in res["error"]


# ── ToolLoop ───────────────────────────────────────────────────────────────

class PlainBackend:
    """No generate_choice/generate_json — ToolLoop should fall back to a single plain generation."""

    def generate(self, prompt=None, messages=None, **kw):
        return "plain answer"


class ScriptedBackend:
    """Grammar-capable fake backend driven by a scripted list of (decision, call) steps."""

    def __init__(self, script):
        self.script = list(script)
        self.step = 0

    def generate_choice(self, choices=None, prompt=None, messages=None, system=""):
        decision, _ = self.script[min(self.step, len(self.script) - 1)]
        return decision

    def generate_json(self, schema=None, prompt=None, messages=None, system="", n_new=256):
        _, call = self.script[min(self.step, len(self.script) - 1)]
        self.step += 1
        return call

    def generate(self, prompt=None, messages=None, **kw):
        return "final answer text"


def test_tool_loop_no_grammar_backend_falls_back():
    reg = ToolRegistry()
    loop = ToolLoop(PlainBackend(), reg)
    res = loop.run("do something")
    assert res["final_answer"] == "plain answer"
    assert res["steps"] == 0
    assert res["tool_calls"] == []


def test_tool_loop_calls_tool_then_finishes(tmp_path):
    (tmp_path / "f.txt").write_text("content", encoding="utf-8")
    reg = build_default_tools(workspace_root=str(tmp_path)).subset(["read_file"])
    backend = ScriptedBackend([
        ("CALL_TOOL", {"tool": "read_file", "args": {"path": "f.txt"}}),
        ("FINAL_ANSWER", None),
    ])
    loop = ToolLoop(backend, reg, max_steps=4)
    res = loop.run("read f.txt")
    assert res["success"] is True
    assert len(res["tool_calls"]) == 1
    assert res["tool_calls"][0]["tool"] == "read_file"
    assert res["tool_calls"][0]["result"]["success"] is True
    assert res["final_answer"] == "final answer text"


def test_tool_loop_dedup_prevents_infinite_repeat(tmp_path):
    (tmp_path / "f.txt").write_text("content", encoding="utf-8")
    reg = build_default_tools(workspace_root=str(tmp_path)).subset(["read_file"])
    # Backend that always tries to call the exact same tool+args — should be deduped, not
    # re-executed forever, and the loop must still terminate within max_steps.
    backend = ScriptedBackend([("CALL_TOOL", {"tool": "read_file", "args": {"path": "f.txt"}})])
    loop = ToolLoop(backend, reg, max_steps=3)
    res = loop.run("read f.txt repeatedly")
    assert res["steps"] == 3
    # Only executed once despite being "requested" every step — the rest were deduped.
    assert len(res["tool_calls"]) == 1


def test_tool_loop_unknown_tool_in_call_treated_as_final(tmp_path):
    reg = build_default_tools(workspace_root=str(tmp_path)).subset(["read_file"])
    backend = ScriptedBackend([("CALL_TOOL", None)])  # no "tool" key -> parse failure
    loop = ToolLoop(backend, reg, max_steps=3)
    res = loop.run("do something ambiguous")
    assert res["final_answer"] == "final answer text"


def test_tool_loop_forced_first_call(tmp_path):
    (tmp_path / "f.txt").write_text("hello world", encoding="utf-8")
    reg = build_default_tools(workspace_root=str(tmp_path)).subset(["read_file"])
    backend = ScriptedBackend([("FINAL_ANSWER", None)])
    loop = ToolLoop(backend, reg, max_steps=3)
    res = loop.run(
        "summarize f.txt",
        forced_first_call={"tool": "read_file", "args": {"path": "f.txt"}},
    )
    assert len(res["tool_calls"]) == 1
    assert res["tool_calls"][0]["tool"] == "read_file"
    assert res["tool_calls"][0]["result"]["result"] == "hello world"


def test_tool_loop_context_budget_forces_final_answer(tmp_path):
    reg = build_default_tools(workspace_root=str(tmp_path)).subset(["read_file"])
    backend = ScriptedBackend([("CALL_TOOL", {"tool": "read_file", "args": {"path": "nope.txt"}})])
    # Tiny budget forces FINAL_ANSWER immediately without ever consulting generate_choice.
    loop = ToolLoop(backend, reg, max_steps=5, context_budget_tokens=1)
    res = loop.run("a" * 100)
    assert res["steps"] == 1
    assert res["tool_calls"] == []
