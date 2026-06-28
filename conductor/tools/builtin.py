"""A small set of read-only builtin tools for v0.

These exist to exercise the provider-agnostic tool-use loop end to end without
touching anything stateful - they are pure/read-only by construction, the same
"read-only by design" stance as hl-read's MCP server. A deliberately
``dangerous`` example tool is included too, so the sandbox gate has something to
refuse in v0 (and something real to route through Proxmox in v1).
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict

from .registry import Tool, ToolSpec


def _now(args: Dict[str, Any]) -> str:
    """Current UTC time in ISO-8601. Read-only."""
    # timezone-aware UTC; no local-clock ambiguity in traces.
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _echo(args: Dict[str, Any]) -> str:
    """Echo back the provided text. Read-only; useful as a loop smoke test."""
    return str(args.get("text", ""))


def _add(args: Dict[str, Any]) -> float:
    """Add two numbers. Pure; a tiny deterministic computation tool."""
    return float(args["a"]) + float(args["b"])


def _run_shell(args: Dict[str, Any]) -> str:
    """A *dangerous* tool: would run a shell command. Blocked without a sandbox.

    In v0 this never executes - it exists so the sandbox gate has a real
    dangerous tool to refuse. In v1 it is dispatched into a Proxmox LXC.
    """
    raise RuntimeError("unreachable in v0: dangerous tools are gated by the registry")


NOW = Tool(
    spec=ToolSpec(
        name="now",
        description="Return the current UTC time in ISO-8601 format. Takes no arguments.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    handler=_now,
)

ECHO = Tool(
    spec=ToolSpec(
        name="echo",
        description="Echo back the given text verbatim.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Text to echo back."}},
            "required": ["text"],
            "additionalProperties": False,
        },
    ),
    handler=_echo,
)

ADD = Tool(
    spec=ToolSpec(
        name="add",
        description="Add two numbers and return their sum.",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First addend."},
                "b": {"type": "number", "description": "Second addend."},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    ),
    handler=_add,
)

# Declared but gated: marks the safety boundary that v1's Proxmox executor fills.
RUN_SHELL = Tool(
    spec=ToolSpec(
        name="run_shell",
        description=(
            "Run a shell command and return its output. DANGEROUS: must execute "
            "inside an isolated sandbox."
        ),
        parameters={
            "type": "object",
            "properties": {"command": {"type": "string", "description": "Command to run."}},
            "required": ["command"],
            "additionalProperties": False,
        },
    ),
    handler=_run_shell,
    dangerous=True,
)

# Dangerous: roll the sandbox back to a snapshot token (handled by SandboxExecutor).
SANDBOX_ROLLBACK = Tool(
    spec=ToolSpec(
        name="sandbox_rollback",
        description=(
            "Roll the sandbox back to a snapshot. Pass the token returned by a "
            "previous run_shell call, or omit it to revert the most recent "
            "run_shell. DANGEROUS: only valid inside a sandbox."
        ),
        parameters={
            "type": "object",
            "properties": {
                "snapshot": {
                    "type": "string",
                    "description": "Snapshot token to restore (optional; defaults to the last run_shell).",
                }
            },
            "additionalProperties": False,
        },
    ),
    handler=_run_shell,  # inert in-process; real handling is in SandboxExecutor
    dangerous=True,
)

# The default read-only toolset for v0 demos and tests.
READONLY_TOOLS = [NOW, ECHO, ADD]

# The sandboxed-shell toolset for v1 (requires a Sandbox wired into the registry).
SANDBOX_TOOLS = [RUN_SHELL, SANDBOX_ROLLBACK]
