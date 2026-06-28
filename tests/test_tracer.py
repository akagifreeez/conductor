"""Tracer writes well-formed, ordered, crash-durable JSONL."""
from __future__ import annotations

import json

from conductor.backends.base import AssistantTurn, ToolCall, ToolResult, Usage
from conductor.tracer import Tracer


def _read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def test_events_written_in_order_and_flushed(tmp_path):
    tr = Tracer("r1", trace_dir=str(tmp_path))
    tr.run_start(provider="local", model="m", task="t", system="s")
    tr.llm_request(step=1, provider="local", model="m", n_messages=1, n_tools=0)
    usage = Usage(prompt_tokens=3, completion_tokens=1, model="m", backend="local")
    turn = AssistantTurn(text="hi", tool_calls=[], stop_reason="end", usage=usage)
    # readable before close() -> proves flush-as-you-go (crash durability)
    tr.llm_response(step=1, turn=turn)
    rows_before_close = _read(tr.path)
    tr.run_end(status="completed", steps=1, final_text="hi")
    tr.close()

    assert len(rows_before_close) == 3
    rows = _read(tr.path)
    assert [r["kind"] for r in rows] == [
        "run_start", "llm_request", "llm_response", "run_end",
    ]
    assert [r["seq"] for r in rows] == [1, 2, 3, 4]


def test_tool_events_serialize_call_and_result(tmp_path):
    with Tracer("r2", trace_dir=str(tmp_path)) as tr:
        tr.tool_call(step=2, call=ToolCall("c1", "now", {"a": 1}))
        tr.tool_result(step=2, result=ToolResult("c1", "now", "out", is_error=True))
    rows = _read(tr.path)
    call, result = rows[0], rows[1]
    assert call["kind"] == "tool_call" and call["name"] == "now" and call["arguments"] == {"a": 1}
    assert result["kind"] == "tool_result" and result["is_error"] is True


def test_usage_embedded_in_llm_response(tmp_path):
    with Tracer("r3", trace_dir=str(tmp_path)) as tr:
        usage = Usage(prompt_tokens=10, completion_tokens=2, model="claude-opus-4-8", backend="anthropic")
        tr.llm_response(
            step=1,
            turn=AssistantTurn(text="x", tool_calls=[ToolCall("i", "now", {})],
                               stop_reason="tool_use", usage=usage),
        )
    row = _read(tr.path)[0]
    assert row["usage"]["backend"] == "anthropic"
    assert row["usage"]["prompt_tokens"] == 10
    assert row["tool_calls"][0]["name"] == "now"
