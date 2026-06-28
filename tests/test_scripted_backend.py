"""The offline scripted backend behaves like a real AgentBackend."""
from __future__ import annotations

from conductor.backends.base import Message, ToolSpec
from conductor.backends.scripted import ScriptedBackend, ScriptedTurn


def _spec():
    return [ToolSpec(name="now", description="x", parameters={"type": "object"})]


def test_tool_use_turn_then_final():
    be = ScriptedBackend(
        [ScriptedTurn(tool_calls=[("now", {})]), ScriptedTurn(text="final")],
        name="m", backend="anthropic",
    )
    t1 = be.step(system="s", messages=[Message(role="user", text="hi")], tools=_spec())
    assert t1.stop_reason == "tool_use"
    assert len(t1.tool_calls) == 1
    assert t1.tool_calls[0].name == "now"
    assert t1.tool_calls[0].call_id  # non-empty, stable id

    t2 = be.step(system="s", messages=[Message(role="user", text="hi")], tools=_spec())
    assert t2.stop_reason == "end"
    assert t2.text == "final"
    assert not t2.tool_calls


def test_usage_is_populated_and_labeled():
    be = ScriptedBackend([ScriptedTurn(text="hello world")], name="modelX", backend="local")
    t = be.step(system="sys", messages=[Message(role="user", text="prompt")], tools=[])
    u = t.usage
    assert u.backend == "local"
    assert u.model == "modelX"
    assert u.prompt_tokens > 0
    assert u.completion_tokens > 0
    assert u.estimated is True


def test_exhausted_script_returns_terminal_turn():
    be = ScriptedBackend([ScriptedTurn(text="only")], name="m", backend="scripted")
    be.step(system="", messages=[], tools=[])  # consume the one turn
    extra = be.step(system="", messages=[], tools=[])  # past the end
    assert extra.stop_reason == "end"
    assert extra.text == ""


def test_call_ids_unique_across_steps():
    be = ScriptedBackend(
        [ScriptedTurn(tool_calls=[("now", {})]), ScriptedTurn(tool_calls=[("now", {})])],
        name="m", backend="scripted",
    )
    a = be.step(system="", messages=[], tools=[]).tool_calls[0].call_id
    b = be.step(system="", messages=[], tools=[]).tool_calls[0].call_id
    assert a != b
