# tests/test_verify_code.py
"""
Unit tests for chakra.agent.verify_code() and the targeted SEARCH/REPLACE
patch retry path wired into KimiAgent.self_debug_loop.
"""

from chakra.agent import KimiAgent, apply_search_replace_block, verify_code


def test_verify_code_valid_code():
    code = "def add(a, b):\n    return a + b\n\nprint(add(2, 3))\n"
    result = verify_code(code)
    assert result["ok"] is True
    assert result["syntax_error"] is None
    assert result["syntax_line"] is None
    assert result["security_issues"] == []
    assert result["has_critical_security"] is False


def test_verify_code_syntax_error():
    # Missing closing paren / colon -> invalid syntax
    code = "def broken(:\n    pass\n"
    result = verify_code(code)
    assert result["ok"] is False
    assert result["syntax_error"] is not None
    assert result["syntax_line"] == 1


def test_verify_code_never_raises_on_garbage_input():
    # Should not raise even on wildly malformed input.
    result = verify_code("\x00\x01 not python at all {{{")
    assert isinstance(result, dict)
    assert result["ok"] is False


def test_verify_code_hardcoded_secret_is_critical():
    # InfoSecAuditor's CRED_PATTERN scores variable names containing
    # "secret"/"private"/"jwt"/"token" as HIGH severity (see chakra/security.py
    # _SecurityASTVisitor.visit_Assign) — confirmed by reading the source before
    # writing this assertion, rather than guessing.
    code = "SECRET_KEY = 'supersecret123'\n"
    result = verify_code(code)
    assert result["ok"] is True or result["security_issues"]  # sanity: audit actually ran
    assert len(result["security_issues"]) >= 1
    assert any(v["severity"] == "HIGH" for v in result["security_issues"])
    assert result["has_critical_security"] is True
    assert result["ok"] is False  # HIGH security finding must fail verify_code


def test_verify_code_non_critical_finding_does_not_block_ok():
    # A MEDIUM-only finding (credential var name without secret/private/jwt/token)
    # should not flip has_critical_security, so ok stays True.
    code = "DB_PASS = 'hunter2xx'\n"
    result = verify_code(code)
    if result["security_issues"]:
        assert not any(v["severity"] == "HIGH" for v in result["security_issues"])
        assert result["has_critical_security"] is False
        assert result["ok"] is True


class ScriptedPatchModel:
    """
    Scripted fake model (same pattern as TruncatedModel in
    tests/test_new_enhancements.py): responds differently based on call count
    rather than actually generating anything.

    Attempt 1 and 2: return code that raises AttributeError (math.sqrtt does
    not exist) — a "localized" failure per _is_localized_diagnosis.
    Attempt 3: once self_debug_loop has escalated to the targeted-patch prompt
    (attempt >= 2 failure with a localized diagnosis), return a SEARCH/REPLACE
    block matching chakra.agent._build_patch_prompt's expected format instead
    of a full script. This can only succeed via apply_search_replace_block —
    the raw text is not valid Python on its own (it contains literal
    '<<<<<<< SEARCH' markers), so naive full-regeneration extraction of this
    response would fail sanitize_and_synthesize_code's AST check and could
    never execute successfully in the sandbox.
    """

    BROKEN_CODE = "```python\nimport math\nprint(math.sqrtt(4))\n```"
    PATCH_RESPONSE = (
        "```\n"
        "<<<<<<< SEARCH\n"
        "print(math.sqrtt(4))\n"
        "=======\n"
        "print(math.sqrt(4))\n"
        ">>>>>>> REPLACE\n"
        "```"
    )

    def __init__(self):
        self.calls = 0
        self.prompts = []

    def __call__(self, prompt):
        self.calls += 1
        self.prompts.append(prompt)
        if self.calls <= 2:
            return self.BROKEN_CODE
        return self.PATCH_RESPONSE

    def generate(self, prompt, n_new=512, system=""):
        return self.__call__(prompt)


def test_self_debug_loop_uses_targeted_patch_to_recover():
    model = ScriptedPatchModel()
    agent = KimiAgent(model=model)

    res = agent.self_debug_loop(
        task_prompt="Print the square root of 4 using math.sqrt",
        max_retries=3,
        gen_tokens=64,
    )

    assert res["success"] is True
    assert res["attempts"] == 3
    assert len(res["history"]) == 3
    assert "2.0" in res["stdout"]

    # Attempts 1 and 2 were full-regeneration (broken) attempts, not patches.
    assert res["history"][0]["patch_applied"] is False
    assert res["history"][1]["patch_applied"] is False
    # Attempt 3 succeeded specifically via the SEARCH/REPLACE patch path.
    assert res["history"][2]["patch_applied"] is True
    assert "math.sqrt(4)" in res["history"][2]["code"]
    assert "sqrtt" not in res["history"][2]["code"]

    # The prompt sent for attempt 3 must be the targeted-patch prompt, i.e. it
    # asked for a SEARCH/REPLACE block rather than a full rewrite.
    patch_prompt = model.prompts[2]
    assert "SEARCH" in patch_prompt
    assert "REPLACE" in patch_prompt

    # Independently confirm apply_search_replace_block is what turned the raw
    # SEARCH/REPLACE response into the working code (proving the patch path,
    # not some other mechanism, produced the fix): applying it manually to the
    # broken code from attempt 2 reproduces attempt 3's code exactly.
    broken_code = res["history"][1]["code"]
    search_block = "print(math.sqrtt(4))"
    replace_block = "print(math.sqrt(4))"
    manual_patched, applied = apply_search_replace_block(broken_code, search_block, replace_block)
    assert applied is True
    assert manual_patched == res["history"][2]["code"]

    # And proves the raw scripted patch response could NOT have succeeded via
    # naive full-regeneration extraction (no apply_search_replace_block at all):
    naive_extracted = agent.sanitize_and_synthesize_code(
        ScriptedPatchModel.PATCH_RESPONSE, "Print the square root of 4 using math.sqrt"
    )
    assert "<<<<<<<" in naive_extracted or "SEARCH" in naive_extracted


def test_self_debug_loop_falls_back_to_full_regen_when_patch_does_not_match():
    """
    If the model's SEARCH/REPLACE block doesn't match anything in last_code,
    apply_search_replace_block returns success=False and self_debug_loop must
    fall back to treating the raw response like a normal (failing) attempt
    rather than crashing.
    """

    class NonMatchingPatchModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, prompt):
            self.calls += 1
            if self.calls <= 2:
                return "```python\nimport math\nprint(math.sqrtt(4))\n```"
            # SEARCH block that will never match last_code's actual content.
            return (
                "```\n"
                "<<<<<<< SEARCH\n"
                "this line does not exist anywhere\n"
                "=======\n"
                "print(math.sqrt(4))\n"
                ">>>>>>> REPLACE\n"
                "```"
            )

        def generate(self, prompt, n_new=512, system=""):
            return self.__call__(prompt)

    agent = KimiAgent(model=NonMatchingPatchModel())
    res = agent.self_debug_loop(
        task_prompt="Print the square root of 4 using math.sqrt",
        max_retries=3,
        gen_tokens=64,
    )

    # Never raises; degrades gracefully to a failed (non-crashing) result.
    assert res["success"] is False
    assert len(res["history"]) == 3
    assert res["history"][2]["patch_applied"] is False


class _ScriptedGrammarBackend:
    """Fake engine-C-shaped backend (has generate_choice/generate_json/generate) so the new
    workspace-aware first-attempt tool loop in self_debug_loop can be tested without a real
    GGUF model."""

    def __init__(self, decisions, calls, final_answer):
        self.decisions = list(decisions)
        self.calls = list(calls)
        self.final_answer = final_answer
        self._decision_i = 0
        self._call_i = 0

    def generate_choice(self, choices=None, prompt=None, messages=None, system=""):
        i = min(self._decision_i, len(self.decisions) - 1)
        self._decision_i += 1
        return self.decisions[i]

    def generate_json(self, schema=None, prompt=None, messages=None, system="", n_new=256):
        i = min(self._call_i, len(self.calls) - 1)
        self._call_i += 1
        return self.calls[i]

    def generate(self, prompt=None, messages=None, **kw):
        return self.final_answer


def test_self_debug_loop_first_attempt_uses_tools_when_workspace_root_given(tmp_path):
    (tmp_path / "math_utils.py").write_text("def square(x):\n    return x * x\n", encoding="utf-8")

    backend = _ScriptedGrammarBackend(
        decisions=["CALL_TOOL", "FINAL_ANSWER"],
        calls=[{"tool": "read_file", "args": {"path": "math_utils.py"}}],
        final_answer="```python\nfrom math_utils import square\nprint(square(3))\n```",
    )
    agent = KimiAgent(model=backend)
    res = agent.self_debug_loop(
        task_prompt="write a script that imports and uses square() from math_utils.py",
        max_retries=1,
        gen_tokens=64,
        workspace_root=str(tmp_path),
    )

    assert res["history"][0]["tool_calls"][0]["tool"] == "read_file"
    assert "from math_utils import square" in res["code"]


def test_self_debug_loop_without_workspace_root_does_not_use_tools():
    """workspace_root=None (today's default everywhere except the CLI's /code path) must keep
    using blind generation — no behavior change for existing callers."""
    class PlainModel:
        def __call__(self, prompt):
            return "```python\nprint('hi')\n```"

        def generate(self, prompt, n_new=512, system=""):
            return self.__call__(prompt)

    agent = KimiAgent(model=PlainModel())
    res = agent.self_debug_loop(task_prompt="print hi", max_retries=1, gen_tokens=32)
    assert res["history"][0]["tool_calls"] == []
