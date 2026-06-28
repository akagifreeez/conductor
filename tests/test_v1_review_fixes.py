"""Regression tests pinning the fixes from the v1 adversarial review."""
from __future__ import annotations

import json

import pytest

from conductor.backends.scripted import ScriptedBackend, ScriptedTurn
from conductor.orchestrator import Orchestrator
from conductor.replay import replay_trace
from conductor.sandbox import SandboxExecutor, SubprocessSandbox
from conductor.tools import READONLY_TOOLS, SANDBOX_TOOLS, ToolRegistry


def _sandbox_registry():
    reg = ToolRegistry()
    for t in SANDBOX_TOOLS:
        reg.register(t)
    return reg


# --- #4: run_end stays the terminal trace event even with a sandbox ------

def test_run_end_is_terminal_with_sandbox(tmp_path):
    be = ScriptedBackend([ScriptedTurn(text="done")], name="m", backend="scripted")
    res = Orchestrator(
        be, _sandbox_registry(), run_id="se", trace_dir=str(tmp_path),
        sandbox=SubprocessSandbox(),
    ).run("noop")
    rows = [json.loads(l) for l in open(res.trace_path, encoding="utf-8")]
    kinds = [r["kind"] for r in rows]
    assert kinds[-1] == "run_end"                 # terminal, after teardown
    assert "sandbox" in kinds
    # teardown is recorded BEFORE run_end
    teardown_idx = max(i for i, r in enumerate(rows) if r["kind"] == "sandbox" and r["event"] == "teardown")
    assert teardown_idx < len(rows) - 1


def test_run_end_terminal_on_error_with_sandbox(tmp_path):
    class Boom(ScriptedBackend):
        def step(self, **kw):
            raise RuntimeError("boom")

    be = Boom([ScriptedTurn(text="x")], name="m", backend="scripted")
    orch = Orchestrator(be, _sandbox_registry(), run_id="seerr",
                        trace_dir=str(tmp_path), sandbox=SubprocessSandbox())
    with pytest.raises(RuntimeError):
        orch.run("noop")
    rows = [json.loads(l) for l in open(str(tmp_path / "run-seerr.jsonl"), encoding="utf-8")]
    assert rows[-1]["kind"] == "run_end"
    assert rows[-1]["status"] == "error"
    # sandbox was still torn down (teardown event present before run_end)
    assert any(r["kind"] == "sandbox" and r["event"] == "teardown" for r in rows)


# --- #1: no-arg sandbox_rollback reverts the most recent run_shell --------

def test_noarg_rollback_reverts_last_run_shell():
    with SubprocessSandbox(seed_files={"a.txt": "1"}) as sb:
        ex = SandboxExecutor(sb)
        reg = _sandbox_registry()
        reg.set_sandbox(ex)
        # delete everything inside the box via run_shell
        import sys
        delete = (f'"{sys.executable}" -c "import glob,os,shutil;'
                  "[shutil.rmtree(p) if os.path.isdir(p) else os.remove(p) for p in glob.glob(chr(42))]\"")
        reg.execute("c1", "run_shell", {"command": delete})
        assert sb.list_files() == []                  # destroyed
        out = reg.execute("c2", "sandbox_rollback", {})  # no token -> revert last
        assert out.is_error is False
        assert sb.list_files() == ["a.txt"]           # restored


# --- #12: orchestrator restores the registry's prior sandbox, not None ----

def test_registry_prior_sandbox_restored_after_run(tmp_path):
    reg = _sandbox_registry()
    sentinel = SandboxExecutor(SubprocessSandbox())   # a pre-existing executor
    reg.set_sandbox(sentinel)
    be = ScriptedBackend([ScriptedTurn(text="done")], name="m", backend="scripted")
    Orchestrator(be, reg, run_id="rest", trace_dir=str(tmp_path),
                 sandbox=SubprocessSandbox()).run("noop")
    assert reg._sandbox is sentinel                   # restored, not clobbered to None


# --- #3: replay raises a clear ValueError on a malformed trace -----------

def test_replay_raises_valueerror_on_malformed_llm_response(tmp_path):
    bad = tmp_path / "bad.jsonl"
    rows = [
        {"run_id": "x", "seq": 1, "kind": "run_start", "provider": "p", "model": "m",
         "task": "t", "system": "s", "tools": []},
        {"run_id": "x", "seq": 2, "kind": "llm_response", "text": "", "stop_reason": "end",
         "tool_calls": []},  # missing "usage" -> incomplete
    ]
    bad.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        replay_trace(str(bad), run_id="r", trace_dir=str(tmp_path))


# --- #11: duplicate call_ids replay in recorded FIFO order ----------------

def test_replay_duplicate_call_ids_fifo(tmp_path):
    # Hand-build a trace where the same call_id appears twice with different
    # recorded outputs; replay must serve them in order, not last-wins.
    rid = "dup"
    rows = [
        {"run_id": rid, "seq": 1, "kind": "run_start", "provider": "scripted", "model": "m",
         "task": "t", "system": "s", "tools": [{"name": "echo", "description": "d",
                                                "parameters": {"type": "object"}}]},
    ]
    usage = {"prompt_tokens": 1, "completion_tokens": 1, "model": "m", "backend": "scripted",
             "latency_ms": 0.0, "estimated": True}
    for i, out in enumerate(["FIRST", "SECOND"]):
        rows.append({"run_id": rid, "seq": 10 + i, "kind": "llm_response", "text": "",
                     "stop_reason": "tool_use",
                     "tool_calls": [{"call_id": "dupe", "name": "echo", "arguments": {}}],
                     "usage": usage, "extra_usages": []})
        rows.append({"run_id": rid, "seq": 20 + i, "kind": "tool_call", "call_id": "dupe",
                     "name": "echo", "arguments": {}})
        rows.append({"run_id": rid, "seq": 30 + i, "kind": "tool_result", "call_id": "dupe",
                     "name": "echo", "content": out, "is_error": False})
    rows.append({"run_id": rid, "seq": 99, "kind": "llm_response", "text": "fin",
                 "stop_reason": "end", "tool_calls": [], "usage": usage, "extra_usages": []})
    rows.append({"run_id": rid, "seq": 100, "kind": "run_end", "status": "completed",
                 "steps": 3, "final_text": "fin"})
    trace = tmp_path / f"run-{rid}.jsonl"
    trace.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    res, cmp = replay_trace(str(trace), run_id="dup-rep", trace_dir=str(tmp_path))
    new_rows = [json.loads(l) for l in open(res.trace_path, encoding="utf-8")]
    contents = [r["content"] for r in new_rows if r["kind"] == "tool_result"]
    assert contents == ["FIRST", "SECOND"]   # FIFO, not ["SECOND","SECOND"]
    assert cmp["match"] is True
