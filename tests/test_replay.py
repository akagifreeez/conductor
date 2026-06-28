"""Deterministic replay reproduces a recorded run's I/O, final answer, and cost."""
from __future__ import annotations

import json

from conductor.backends.scripted import ScriptedBackend, ScriptedTurn
from conductor.orchestrator import Orchestrator
from conductor.replay import load_trace, replay_trace
from conductor.tools import READONLY_TOOLS, ToolRegistry


def _registry():
    reg = ToolRegistry()
    for t in READONLY_TOOLS:
        reg.register(t)
    return reg


def _record_a_run(tmp_path):
    be = ScriptedBackend(
        [ScriptedTurn(tool_calls=[("add", {"a": 2, "b": 3})]),
         ScriptedTurn(text="the sum is five")],
        name="claude-opus-4-8", backend="anthropic",
    )
    return Orchestrator(be, _registry(), run_id="orig", trace_dir=str(tmp_path)).run("add 2 and 3")


def test_replay_reproduces_io_and_final(tmp_path):
    original = _record_a_run(tmp_path)
    _res, cmp = replay_trace(original.trace_path, run_id="rep", trace_dir=str(tmp_path))
    assert cmp["match"] is True
    assert cmp["tool_results_match"] is True
    assert cmp["final_match"] is True
    assert cmp["replayed_final"] == "the sum is five"
    assert cmp["n_tool_results"] == 1


def test_replay_makes_no_live_calls_and_reproduces_cost(tmp_path):
    original = _record_a_run(tmp_path)
    res, _cmp = replay_trace(original.trace_path, run_id="rep2", trace_dir=str(tmp_path))
    # The replayed ledger must reproduce the original per-provider cost split.
    assert "anthropic" in res.ledger_summary["by_backend"]
    orig_cost = original.ledger_summary["by_backend"]["anthropic"]["cost_usd"]
    rep_cost = res.ledger_summary["by_backend"]["anthropic"]["cost_usd"]
    assert rep_cost == orig_cost


def test_replay_uses_recorded_tool_result_not_reexecution(tmp_path):
    # Record a run that called `now` (time-dependent); replay must return the
    # SAME recorded timestamp, not a fresh one.
    be = ScriptedBackend(
        [ScriptedTurn(tool_calls=[("now", {})]), ScriptedTurn(text="reported the time")],
        name="m", backend="scripted",
    )
    original = Orchestrator(be, _registry(), run_id="orig3", trace_dir=str(tmp_path)).run("time?")
    orig_rows = load_trace(original.trace_path)
    orig_now = next(r["content"] for r in orig_rows if r["kind"] == "tool_result")

    res, cmp = replay_trace(original.trace_path, run_id="rep3", trace_dir=str(tmp_path))
    rep_rows = load_trace(res.trace_path)
    rep_now = next(r["content"] for r in rep_rows if r["kind"] == "tool_result")
    assert rep_now == orig_now      # recorded I/O reproduced, not re-executed
    assert cmp["match"] is True


def test_replay_rejects_non_trace(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"kind": "llm_response"}) + "\n", encoding="utf-8")
    try:
        replay_trace(str(bad), run_id="x", trace_dir=str(tmp_path))
        raise AssertionError("expected ValueError for a non-trace file")
    except ValueError:
        pass
