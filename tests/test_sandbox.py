"""Sandbox layer: SubprocessSandbox contract, gate wiring, end-to-end rollback."""
from __future__ import annotations

import json
import os
import sys

import pytest

from conductor.backends.scripted import ScriptedBackend, ScriptedTurn
from conductor.orchestrator import Orchestrator
from conductor.sandbox import SandboxExecutor, SubprocessSandbox
from conductor.sandbox.proxmox import ProxmoxSandbox
from conductor.tools import SANDBOX_TOOLS, ToolRegistry
from conductor.tools.registry import SandboxRequiredError

PY = sys.executable  # portable: drive file ops through the running interpreter

_LIST = (
    f'"{PY}" -c "import glob,os;'
    "print(sorted(p.replace(os.sep,chr(47)) for p in glob.glob(chr(42)) if os.path.isfile(p)))\""
)
_DELETE_ALL = (
    f'"{PY}" -c "import glob,os,shutil;'
    "[shutil.rmtree(p) if os.path.isdir(p) else os.remove(p) for p in glob.glob(chr(42))]\""
)
# Delete everything, THEN print the (now empty) listing in the same command.
_DESTROY_AND_LIST = (
    f'"{PY}" -c "import glob,os,shutil;'
    "[shutil.rmtree(p) if os.path.isdir(p) else os.remove(p) for p in glob.glob(chr(42))];"
    "print(sorted(p.replace(os.sep,chr(47)) for p in glob.glob(chr(42)) if os.path.isfile(p)))\""
)


def _sandbox():
    return SubprocessSandbox(seed_files={"data.txt": "important", "b.txt": "keep"})


# --- SubprocessSandbox direct contract -----------------------------------

def test_snapshot_exec_rollback_cycle():
    with _sandbox() as sb:
        assert sb.list_files() == ["b.txt", "data.txt"]
        token = sb.snapshot("before")
        res = sb.exec(_DELETE_ALL)
        assert res.ok, res.stderr
        assert sb.list_files() == []          # destruction happened in the box
        sb.rollback(token)
        assert sb.list_files() == ["b.txt", "data.txt"]  # fully restored


def test_exec_runs_in_box_not_host_cwd(tmp_path, monkeypatch):
    # A sentinel in the real cwd must be untouched by an in-sandbox delete-all.
    monkeypatch.chdir(tmp_path)
    sentinel = tmp_path / "HOST_FILE.txt"
    sentinel.write_text("host", encoding="utf-8")
    with _sandbox() as sb:
        sb.exec(_DELETE_ALL)
        assert sb.list_files() == []
    assert sentinel.exists()  # host cwd never was the exec target


def test_rollback_unknown_token_raises():
    with _sandbox() as sb:
        with pytest.raises(ValueError):
            sb.rollback("nope")


def test_exec_before_setup_raises():
    sb = SubprocessSandbox()
    with pytest.raises(RuntimeError):
        sb.exec("echo hi")


# --- gate wiring via SandboxExecutor -------------------------------------

def test_registry_blocks_dangerous_without_sandbox():
    reg = ToolRegistry()
    for t in SANDBOX_TOOLS:
        reg.register(t)
    res = reg.execute("c", "run_shell", {"command": "echo hi"})
    assert res.is_error and "sandbox" in res.content.lower()


def test_executor_run_shell_returns_snapshot_token():
    with _sandbox() as sb:
        ex = SandboxExecutor(sb)
        reg = ToolRegistry()
        for t in SANDBOX_TOOLS:
            reg.register(t)
        reg.set_sandbox(ex)
        res = reg.execute("c1", "run_shell", {"command": _LIST})
        payload = json.loads(res.content)
        assert "snapshot" in payload and payload["exit_code"] == 0
        assert "data.txt" in payload["stdout"]


def test_executor_rejects_unknown_dangerous_tool():
    from conductor.tools.registry import Tool
    from conductor.backends.base import ToolSpec
    with _sandbox() as sb:
        ex = SandboxExecutor(sb)
        evil = Tool(spec=ToolSpec(name="format_disk", description="x", parameters={"type": "object"}),
                    handler=lambda a: None, dangerous=True)
        with pytest.raises(SandboxRequiredError):
            ex(evil, {})


# --- end-to-end: dangerous command contained + rolled back ----------------

def test_orchestrator_sandbox_rollback_end_to_end(tmp_path):
    reg = ToolRegistry()
    for t in SANDBOX_TOOLS:
        reg.register(t)
    # ls -> destroy(deletes + lists empty) -> rollback(no token, reverts last
    # run_shell) -> ls(restored). No hardcoded snapshot token.
    script = [
        ScriptedTurn(tool_calls=[("run_shell", {"command": _LIST})]),
        ScriptedTurn(tool_calls=[("run_shell", {"command": _DESTROY_AND_LIST})]),
        ScriptedTurn(tool_calls=[("sandbox_rollback", {})]),
        ScriptedTurn(tool_calls=[("run_shell", {"command": _LIST})]),
        ScriptedTurn(text="done: destruction was contained and rolled back"),
    ]
    be = ScriptedBackend(script, name="m", backend="scripted")
    res = Orchestrator(
        be, reg, run_id="sbx", trace_dir=str(tmp_path), max_steps=10,
        sandbox=SubprocessSandbox(seed_files={"data.txt": "important"}),
    ).run("clean up then undo")

    assert res.status == "completed"
    rows = [json.loads(l) for l in open(res.trace_path, encoding="utf-8")]
    # sandbox lifecycle is traced
    kinds = [r["kind"] for r in rows]
    assert "sandbox" in kinds
    sandbox_events = [r["event"] for r in rows if r["kind"] == "sandbox"]
    assert "setup" in sandbox_events and "teardown" in sandbox_events
    # tool outputs tell the story. Keep only the `_LIST` outputs (their stdout is
    # a python list repr starting with "[") so the delete command's empty stdout
    # doesn't get counted as a listing.
    list_outputs = []
    for r in rows:
        if r["kind"] == "tool_result" and r["content"].startswith("{") and "stdout" in r["content"]:
            stdout = json.loads(r["content"]).get("stdout", "")
            if stdout.strip().startswith("["):
                list_outputs.append(stdout)
    assert len(list_outputs) == 3
    assert "data.txt" in list_outputs[0]      # before
    assert "data.txt" not in list_outputs[1]  # after delete (contained in the box)
    assert "data.txt" in list_outputs[2]      # after rollback (restored)


# --- ProxmoxSandbox: fails fast & clean when unconfigured (no network) -----

def test_proxmox_sandbox_unconfigured_raises():
    sb = ProxmoxSandbox(vmid=999, node="pve")
    # No proxmoxer installed / no creds -> a clear RuntimeError, never a silent hang.
    with pytest.raises(RuntimeError):
        sb.setup()
