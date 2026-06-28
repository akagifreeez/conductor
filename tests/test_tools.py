"""Tool registry: execution, error handling, and the sandbox gate."""
from __future__ import annotations

from conductor.tools import READONLY_TOOLS, RUN_SHELL, ToolRegistry
from conductor.tools.registry import SandboxRequiredError, Tool
from conductor.backends.base import ToolSpec


def _registry(*tools: Tool) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


def test_register_and_specs():
    reg = _registry(*READONLY_TOOLS)
    names = {s.name for s in reg.specs()}
    assert names == {"now", "echo", "add"}


def test_duplicate_registration_rejected():
    reg = ToolRegistry()
    reg.register(READONLY_TOOLS[0])
    try:
        reg.register(READONLY_TOOLS[0])
    except ValueError:
        return
    raise AssertionError("duplicate registration should raise ValueError")


def test_execute_readonly_tool():
    reg = _registry(*READONLY_TOOLS)
    res = reg.execute("c1", "add", {"a": 2, "b": 3})
    assert res.is_error is False
    assert res.call_id == "c1"
    assert res.content == "5.0"


def test_echo_serializes_to_text():
    reg = _registry(*READONLY_TOOLS)
    res = reg.execute("c2", "echo", {"text": "hi"})
    assert res.content == "hi"
    assert res.is_error is False


def test_unknown_tool_returns_error_result_not_raise():
    reg = _registry(*READONLY_TOOLS)
    res = reg.execute("c3", "does_not_exist", {})
    assert res.is_error is True
    assert "unknown tool" in res.content


def test_handler_exception_becomes_error_result():
    reg = _registry(*READONLY_TOOLS)
    # add with missing args -> KeyError inside handler -> error result, no crash.
    res = reg.execute("c4", "add", {"a": 1})
    assert res.is_error is True
    assert "Error running 'add'" in res.content


def test_dangerous_tool_blocked_without_sandbox():
    reg = _registry(RUN_SHELL)
    res = reg.execute("c5", "run_shell", {"command": "rm -rf /"})
    assert res.is_error is True
    assert "Blocked" in res.content
    assert "sandbox" in res.content.lower()


def test_dangerous_tool_dispatched_through_sandbox():
    calls = {}

    def fake_sandbox(tool, args):
        calls["tool"] = tool.spec.name
        calls["args"] = args
        return "ran-in-sandbox"

    reg = ToolRegistry(sandbox=fake_sandbox)
    reg.register(RUN_SHELL)
    res = reg.execute("c6", "run_shell", {"command": "echo hi"})
    assert res.is_error is False
    assert res.content == "ran-in-sandbox"
    assert calls == {"tool": "run_shell", "args": {"command": "echo hi"}}


def test_custom_tool_dict_return_is_json_serialized():
    spec = ToolSpec(name="kv", description="returns a dict", parameters={"type": "object"})
    reg = _registry(Tool(spec=spec, handler=lambda a: {"x": 1, "y": [2, 3]}))
    res = reg.execute("c7", "kv", {})
    assert res.content == '{"x": 1, "y": [2, 3]}'
    assert res.is_error is False
