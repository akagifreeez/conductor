"""OpenAI-compatible adapter: translation both ways, no network."""
from __future__ import annotations

import json

from conductor.backends.base import Message, ToolCall, ToolResult, ToolSpec
from conductor.backends.openai_compat import OpenAICompatAdapter


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _FakeSession:
    """Returns queued responses and records the bodies it was sent."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.sent = []

    def post(self, url, headers=None, data=None, timeout=None):
        self.sent.append({"url": url, "headers": headers, "body": json.loads(data)})
        return self._responses.pop(0)


def _adapter(session, **kw):
    return OpenAICompatAdapter("test-model", base_url="http://x/v1", session=session, **kw)


def test_to_openai_messages_shapes():
    msgs = [
        Message(role="user", text="hi"),
        Message(role="assistant", text="thinking", tool_calls=[ToolCall("id1", "now", {})]),
        Message(role="user", tool_results=[ToolResult("id1", "now", "2026-01-01")]),
    ]
    out = OpenAICompatAdapter._to_openai_messages("SYS", msgs)
    assert out[0] == {"role": "system", "content": "SYS"}
    assert out[1] == {"role": "user", "content": "hi"}
    assert out[2]["role"] == "assistant"
    assert out[2]["tool_calls"][0]["id"] == "id1"
    assert out[2]["tool_calls"][0]["function"]["name"] == "now"
    # tool result becomes a role="tool" message paired by id
    assert out[3] == {"role": "tool", "tool_call_id": "id1", "content": "2026-01-01"}


def test_to_openai_tools_shape():
    spec = [ToolSpec(name="add", description="adds", parameters={"type": "object"})]
    out = OpenAICompatAdapter._to_openai_tools(spec)
    assert out[0]["type"] == "function"
    assert out[0]["function"]["name"] == "add"
    assert out[0]["function"]["parameters"] == {"type": "object"}


def test_step_parses_tool_call_with_json_string_arguments():
    payload = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_9",
                            "type": "function",
                            "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'},
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 3},
    }
    be = _adapter(_FakeSession([_FakeResponse(payload)]))
    turn = be.step(system="s", messages=[Message(role="user", text="add")], tools=[])
    assert turn.stop_reason == "tool_use"
    assert turn.tool_calls[0].name == "add"
    assert turn.tool_calls[0].arguments == {"a": 1, "b": 2}
    assert turn.usage.prompt_tokens == 11
    assert turn.usage.completion_tokens == 3
    assert turn.usage.backend == "openai-compat"
    assert turn.usage.estimated is False


def test_step_parses_final_text():
    payload = {
        "choices": [{"message": {"content": "the answer is 3"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 4},
    }
    be = _adapter(_FakeSession([_FakeResponse(payload)]))
    turn = be.step(system="s", messages=[Message(role="user", text="q")], tools=[])
    assert turn.stop_reason == "end"
    assert turn.text == "the answer is 3"
    assert not turn.tool_calls


def test_malformed_tool_arguments_do_not_crash():
    payload = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {"id": "c", "type": "function",
                         "function": {"name": "add", "arguments": "{not json"}}
                    ],
                }
            }
        ]
    }
    be = _adapter(_FakeSession([_FakeResponse(payload)]))
    turn = be.step(system="s", messages=[Message(role="user", text="x")], tools=[])
    # Doesn't raise; surfaces the raw blob so the tool layer can error cleanly.
    assert turn.tool_calls[0].arguments == {"__raw__": "{not json"}


def test_usage_estimated_when_backend_omits_it():
    payload = {"choices": [{"message": {"content": "hello"}}]}  # no usage field
    be = _adapter(_FakeSession([_FakeResponse(payload)]))
    turn = be.step(system="s", messages=[Message(role="user", text="hi there")], tools=[])
    assert turn.usage.estimated is True
    assert turn.usage.prompt_tokens > 0
    assert turn.usage.completion_tokens > 0


def test_local_backend_label_and_no_auth_header():
    payload = {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    sess = _FakeSession([_FakeResponse(payload)])
    be = OpenAICompatAdapter("qwen", base_url="http://x/v1", session=sess, backend="local", api_key="")
    turn = be.step(system="s", messages=[Message(role="user", text="hi")], tools=[])
    assert turn.usage.backend == "local"
    # no Authorization header when key is empty
    assert "Authorization" not in sess.sent[0]["headers"]


def test_tools_included_in_request_body():
    payload = {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    sess = _FakeSession([_FakeResponse(payload)])
    be = _adapter(sess)
    spec = [ToolSpec(name="now", description="d", parameters={"type": "object"})]
    be.step(system="s", messages=[Message(role="user", text="x")], tools=spec)
    body = sess.sent[0]["body"]
    assert body["tools"][0]["function"]["name"] == "now"
    assert body["model"] == "test-model"
