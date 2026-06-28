"""The self-built tool-use loop: end-to-end behavior, accounting, and limits."""
from __future__ import annotations

import json

from token_router.accounting import Ledger

from conductor.backends.scripted import ScriptedBackend, ScriptedTurn
from conductor.orchestrator import Orchestrator
from conductor.pricing import make_pricing
from conductor.tools import READONLY_TOOLS, ToolRegistry


def _registry():
    reg = ToolRegistry()
    for t in READONLY_TOOLS:
        reg.register(t)
    return reg


def test_loop_runs_tool_then_completes(tmp_path):
    be = ScriptedBackend(
        [ScriptedTurn(tool_calls=[("add", {"a": 2, "b": 5})]), ScriptedTurn(text="seven")],
        name="m", backend="anthropic",
    )
    orch = Orchestrator(be, _registry(), run_id="t1", trace_dir=str(tmp_path))
    res = orch.run("add 2 and 5")

    assert res.status == "completed"
    assert res.steps == 2
    assert res.final_text == "seven"


def test_tool_result_is_fed_back_into_history(tmp_path):
    # Capture the messages the backend sees on its second step.
    seen = {}

    class Spy(ScriptedBackend):
        def step(self, *, system, messages, tools, max_tokens=1024, temperature=0.0):
            seen[self._i] = [
                (m.role, bool(m.tool_calls), [r.content for r in m.tool_results])
                for m in messages
            ]
            return super().step(system=system, messages=messages, tools=tools)

    be = Spy(
        [ScriptedTurn(tool_calls=[("add", {"a": 1, "b": 1})]), ScriptedTurn(text="done")],
        name="m", backend="anthropic",
    )
    orch = Orchestrator(be, _registry(), run_id="t2", trace_dir=str(tmp_path))
    orch.run("go")

    # On the second step the history must contain: user task, assistant(tool call),
    # user(tool result "2.0").
    second = seen[1]
    assert second[0][0] == "user"
    assert second[1] == ("assistant", True, [])
    assert second[2][0] == "user"
    assert second[2][2] == ["2.0"]


def test_ledger_splits_cost_per_provider(tmp_path):
    shared = Ledger(pricing=make_pricing())
    script = [ScriptedTurn(tool_calls=[("now", {})]), ScriptedTurn(text="ok")]

    for label, model in [("anthropic", "claude-opus-4-8"), ("local", "qwen2.5:3b-instruct")]:
        be = ScriptedBackend(list(script), name=model, backend=label)
        Orchestrator(
            be, _registry(), run_id=f"r-{label}", trace_dir=str(tmp_path), ledger=shared
        ).run("what time is it")

    s = shared.summary()
    assert set(s["by_backend"]) == {"anthropic", "local"}
    assert s["by_backend"]["anthropic"]["cost_usd"] > 0   # priced
    assert s["by_backend"]["local"]["cost_usd"] == 0.0    # local is free
    assert s["calls"] == 4  # 2 llm calls per provider


def test_max_steps_terminates_without_hanging(tmp_path):
    # A backend that ALWAYS asks for a tool would loop forever without the bound.
    be = ScriptedBackend(
        [ScriptedTurn(tool_calls=[("now", {})]) for _ in range(50)],
        name="m", backend="scripted",
    )
    orch = Orchestrator(be, _registry(), run_id="t3", trace_dir=str(tmp_path), max_steps=3)
    res = orch.run("loop")
    assert res.status == "max_steps"
    assert res.steps == 3


def test_trace_file_is_valid_jsonl_with_expected_events(tmp_path):
    be = ScriptedBackend(
        [ScriptedTurn(tool_calls=[("now", {})]), ScriptedTurn(text="ok")],
        name="m", backend="anthropic",
    )
    res = Orchestrator(be, _registry(), run_id="t4", trace_dir=str(tmp_path)).run("go")

    lines = [json.loads(l) for l in open(res.trace_path, encoding="utf-8")]
    kinds = [e["kind"] for e in lines]
    assert kinds[0] == "run_start"
    assert kinds[-1] == "run_end"
    assert "tool_call" in kinds and "tool_result" in kinds
    # seq is strictly increasing
    seqs = [e["seq"] for e in lines]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


def test_dangerous_tool_blocked_end_to_end(tmp_path):
    from conductor.tools import RUN_SHELL

    reg = ToolRegistry()
    reg.register(RUN_SHELL)
    be = ScriptedBackend(
        [ScriptedTurn(tool_calls=[("run_shell", {"command": "rm -rf /"})]),
         ScriptedTurn(text="i could not run that")],
        name="m", backend="scripted",
    )
    res = Orchestrator(be, reg, run_id="t5", trace_dir=str(tmp_path)).run("delete everything")

    lines = [json.loads(l) for l in open(res.trace_path, encoding="utf-8")]
    tool_results = [e for e in lines if e["kind"] == "tool_result"]
    assert tool_results and tool_results[0]["is_error"] is True
    assert "Blocked" in tool_results[0]["content"]
    # The run still completes gracefully (model gets the error and finishes).
    assert res.status == "completed"
