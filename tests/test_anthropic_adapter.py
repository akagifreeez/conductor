"""Anthropic adapter: translation both ways via an injected fake client.

No real ``anthropic`` SDK or network is used - a fake client is injected through
the ``client=`` constructor arg, exactly the seam that exists for this.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from token_router.models.base import ModelError

from conductor.backends.anthropic_adapter import AnthropicAdapter
from conductor.backends.base import Message, ToolCall, ToolResult, ToolSpec


class _FakeMessages:
    def __init__(self, response, recorder):
        self._response = response
        self._recorder = recorder

    def create(self, **kwargs):
        self._recorder.append(kwargs)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class _FakeClient:
    def __init__(self, response, recorder):
        self.messages = _FakeMessages(response, recorder)


def _resp(content_blocks, input_tokens=12, output_tokens=4):
    return SimpleNamespace(
        content=content_blocks,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def test_to_anthropic_messages_shapes():
    msgs = [
        Message(role="user", text="hi"),
        Message(role="assistant", text="ok", tool_calls=[ToolCall("u1", "now", {"x": 1})]),
        Message(role="user", tool_results=[ToolResult("u1", "now", "2026", is_error=False)]),
    ]
    out = AnthropicAdapter._to_anthropic_messages(msgs)
    assert out[0] == {"role": "user", "content": "hi"}
    assert out[1]["role"] == "assistant"
    assert out[1]["content"][0] == {"type": "text", "text": "ok"}
    assert out[1]["content"][1] == {"type": "tool_use", "id": "u1", "name": "now", "input": {"x": 1}}
    assert out[2]["role"] == "user"
    assert out[2]["content"][0]["type"] == "tool_result"
    assert out[2]["content"][0]["tool_use_id"] == "u1"


def test_to_anthropic_tools_uses_input_schema_key():
    spec = [ToolSpec(name="add", description="adds", parameters={"type": "object"})]
    out = AnthropicAdapter._to_anthropic_tools(spec)
    assert out[0] == {"name": "add", "description": "adds", "input_schema": {"type": "object"}}


def test_step_parses_tool_use_block():
    rec = []
    block = SimpleNamespace(type="tool_use", id="abc", name="now", input={})
    client = _FakeClient(_resp([block]), rec)
    be = AnthropicAdapter("claude-opus-4-8", client=client)
    turn = be.step(system="SYS", messages=[Message(role="user", text="time?")],
                   tools=[ToolSpec(name="now", description="d", parameters={"type": "object"})])
    assert turn.stop_reason == "tool_use"
    assert turn.tool_calls[0].call_id == "abc"
    assert turn.tool_calls[0].name == "now"
    assert turn.usage.backend == "anthropic"
    assert turn.usage.prompt_tokens == 12
    assert turn.usage.completion_tokens == 4
    assert turn.usage.estimated is False
    # system is passed as a top-level param, not a message
    assert rec[0]["system"] == "SYS"


def test_step_parses_text_block_as_final():
    rec = []
    block = SimpleNamespace(type="text", text="the time is now")
    be = AnthropicAdapter("claude-opus-4-8", client=_FakeClient(_resp([block]), rec))
    turn = be.step(system="", messages=[Message(role="user", text="q")], tools=[])
    assert turn.stop_reason == "end"
    assert turn.text == "the time is now"
    assert not turn.tool_calls
    # no system param when system is empty
    assert "system" not in rec[0]


def test_step_normalizes_sdk_error_to_modelerror():
    be = AnthropicAdapter("claude-opus-4-8", client=_FakeClient(RuntimeError("boom"), []))
    with pytest.raises(ModelError):
        be.step(system="", messages=[Message(role="user", text="q")], tools=[])


def test_mixed_text_and_tool_use_blocks():
    rec = []
    blocks = [
        SimpleNamespace(type="text", text="let me check"),
        SimpleNamespace(type="tool_use", id="t2", name="add", input={"a": 1, "b": 2}),
    ]
    be = AnthropicAdapter("claude-opus-4-8", client=_FakeClient(_resp(blocks), rec))
    turn = be.step(system="s", messages=[Message(role="user", text="add")], tools=[])
    assert turn.text == "let me check"
    assert turn.stop_reason == "tool_use"
    assert turn.tool_calls[0].arguments == {"a": 1, "b": 2}
