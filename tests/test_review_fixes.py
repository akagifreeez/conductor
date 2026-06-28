"""Regression tests pinning the fixes from the v0 adversarial review.

Each test corresponds to a confirmed finding, so a future change can't silently
re-introduce the issue.
"""
from __future__ import annotations

import json

from types import SimpleNamespace

from token_router.accounting import Ledger
from token_router.models.base import count_tokens

from conductor.backends.base import Message, ToolCall, ToolResult, estimate_prompt_text
from conductor.backends.anthropic_adapter import AnthropicAdapter
from conductor.backends.openai_compat import OpenAICompatAdapter
from conductor.backends.scripted import ScriptedBackend, ScriptedTurn
from conductor.orchestrator import Orchestrator, conductor_summary
from conductor.pricing import make_pricing
from conductor.tools import READONLY_TOOLS, ToolRegistry


def _registry():
    reg = ToolRegistry()
    for t in READONLY_TOOLS:
        reg.register(t)
    return reg


# --- estimate includes tool I/O (understatement fix) ---------------------

def test_estimate_prompt_text_includes_tool_calls_and_results():
    msgs = [
        Message(role="user", text="hi"),
        Message(role="assistant", text="", tool_calls=[ToolCall("c", "lookup", {"q": "weather in tokyo"})]),
        Message(role="user", tool_results=[ToolResult("c", "lookup", "sunny, 24C, humidity 60%")]),
    ]
    text = estimate_prompt_text("SYS", msgs)
    # tool name, serialized args, and result content all contribute
    assert "lookup" in text
    assert "weather in tokyo" in text
    assert "sunny, 24C" in text
    # strictly more than the naive text-only join (which would miss tool I/O)
    naive = "SYS" + "".join(m.text for m in msgs)
    assert count_tokens(text) > count_tokens(naive)


# --- OpenAI preserves is_error (cross-provider parity) --------------------

def test_openai_tool_result_preserves_is_error_signal():
    msgs = [Message(role="user", tool_results=[ToolResult("c", "run_shell", "Blocked: dangerous", is_error=True)])]
    out = OpenAICompatAdapter._to_openai_messages("", msgs)
    assert out[0]["role"] == "tool"
    assert out[0]["content"].startswith("[tool error]")
    assert "Blocked: dangerous" in out[0]["content"]


def test_openai_tool_result_ok_is_unprefixed():
    msgs = [Message(role="user", tool_results=[ToolResult("c", "now", "2026-01-01", is_error=False)])]
    out = OpenAICompatAdapter._to_openai_messages("", msgs)
    assert out[0]["content"] == "2026-01-01"


# --- Anthropic skips empty assistant content (latent 400 fix) -------------

def test_anthropic_skips_empty_assistant_message():
    msgs = [
        Message(role="user", text="hi"),
        Message(role="assistant", text="", tool_calls=[]),  # no info -> must be dropped
        Message(role="user", text="still here"),
    ]
    out = AnthropicAdapter._to_anthropic_messages(msgs)
    # the empty assistant turn is gone; no message has an empty content list
    assert len(out) == 2
    for m in out:
        assert m["content"] != []


# --- final_text retained on max_steps (loop fix) -------------------------

def test_final_text_retained_when_truncated_with_narration(tmp_path):
    be = ScriptedBackend(
        [ScriptedTurn(text="let me check...", tool_calls=[("now", {})]) for _ in range(5)],
        name="m", backend="scripted",
    )
    res = Orchestrator(be, _registry(), run_id="rx", trace_dir=str(tmp_path), max_steps=2).run("go")
    assert res.status == "max_steps"
    # the narration is NOT lost just because we truncated on a tool-calling turn
    assert res.final_text == "let me check..."


# --- backend exception still terminates the trace (durability fix) --------

def test_backend_exception_emits_run_end_error(tmp_path):
    class Boom(ScriptedBackend):
        def step(self, **kw):
            raise RuntimeError("backend down")

    be = Boom([ScriptedTurn(text="x")], name="m", backend="scripted")
    orch = Orchestrator(be, _registry(), run_id="rerr", trace_dir=str(tmp_path))
    try:
        orch.run("go")
        raise AssertionError("expected the backend error to propagate")
    except RuntimeError:
        pass
    rows = [json.loads(l) for l in open(str(tmp_path / "run-rerr.jsonl"), encoding="utf-8")]
    assert rows[-1]["kind"] == "run_end"
    assert rows[-1]["status"] == "error"


# --- owned ledger is persisted to disk (durability fix) ------------------

def test_owned_ledger_written_to_disk(tmp_path):
    be = ScriptedBackend([ScriptedTurn(text="ok")], name="claude-opus-4-8", backend="anthropic")
    Orchestrator(be, _registry(), run_id="rl", trace_dir=str(tmp_path)).run("hi")
    ledger_file = tmp_path / "ledger-rl.jsonl"
    assert ledger_file.exists()
    rows = [json.loads(l) for l in open(ledger_file, encoding="utf-8")]
    assert rows and rows[0]["backend"] == "anthropic"


# --- conductor_summary drops misleading router-only fields ----------------

def test_conductor_summary_drops_router_only_fields():
    led = Ledger(pricing=make_pricing())
    be = ScriptedBackend([ScriptedTurn(text="ok")], name="claude-opus-4-8", backend="anthropic")
    Orchestrator(be, _registry(), run_id="rs", trace_dir="traces", ledger=led).run("hi")
    s = conductor_summary(led)
    for k in ("kept_local_tasks", "escalated_tasks", "local_keep_rate"):
        assert k not in s
    # the part we DO rely on survives
    assert "by_backend" in s and "anthropic" in s["by_backend"]


# --- run_start records advertised tool specs (replay self-description) ----

def test_run_start_records_tool_specs(tmp_path):
    be = ScriptedBackend([ScriptedTurn(text="ok")], name="m", backend="scripted")
    res = Orchestrator(be, _registry(), run_id="rt", trace_dir=str(tmp_path)).run("hi")
    first = json.loads(open(res.trace_path, encoding="utf-8").readline())
    assert first["kind"] == "run_start"
    names = {t["name"] for t in first["tools"]}
    assert {"now", "echo", "add"} <= names
