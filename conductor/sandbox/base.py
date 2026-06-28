"""The sandbox abstraction - where dangerous tools actually run.

This is the v1 fulfillment of the v0 sandbox *gate*. In v0 a tool marked
``dangerous`` was simply refused; here a ``Sandbox`` executes it in an isolated
environment with snapshot/rollback, so a destructive command runs against the
sandbox and can be reverted - the host is never the execution target.

Two implementations, deliberately at different honesty tiers:

* ``ProxmoxSandbox`` (``conductor.sandbox.proxmox``) - **real OS-level isolation**
  in a Proxmox LXC. This is the actual differentiator and the path you run on a
  real Proxmox node. It is implemented against the Proxmox VE API (mirroring the
  author's proxmoxbot) + SSH for ``pct exec``; it needs a live node and is NOT
  exercised by the offline test suite.

* ``SubprocessSandbox`` (``conductor.sandbox.local``) - an **offline** double that
  runs commands in a throwaway temp directory with a real copy-tree snapshot and
  rollback. It executes real commands and really reverts them, so it proves the
  *gate + snapshot/exec/rollback contract* with no homelab - but it is **NOT a
  security boundary** (a command can still touch absolute paths). It exists for
  tests and the demo, not for containing hostile code.

Keeping these two behind one interface means the orchestrator, the gate, and the
trace are identical whether you run the offline demo or the real LXC.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..tools.registry import SandboxRequiredError, Tool


@dataclass(frozen=True)
class ExecResult:
    """The outcome of running one command in a sandbox."""

    exit_code: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    def to_dict(self) -> Dict[str, Any]:
        return {"exit_code": self.exit_code, "stdout": self.stdout, "stderr": self.stderr}


class Sandbox(abc.ABC):
    """An isolated environment with snapshot/exec/rollback.

    Lifecycle: ``setup()`` once, then any number of ``snapshot()`` / ``exec()`` /
    ``rollback()``, then ``teardown()``. Usable as a context manager.
    ``name``/``kind`` are for tracing and the README's honesty framing
    (e.g. ``kind="proxmox-lxc (real OS isolation)"`` vs
    ``kind="subprocess (NOT a security boundary)"`` - quoting the exact attribute
    values the implementations set).
    """

    name: str
    kind: str

    @abc.abstractmethod
    def setup(self) -> None:
        """Create/start the isolated environment. Must fail fast on misconfig."""

    @abc.abstractmethod
    def snapshot(self, label: str) -> str:
        """Take a snapshot; return an opaque token to pass to ``rollback``."""

    @abc.abstractmethod
    def exec(self, command: str, *, timeout: float = 60.0) -> ExecResult:
        """Run ``command`` inside the sandbox and return its result."""

    @abc.abstractmethod
    def rollback(self, token: str) -> None:
        """Restore the sandbox to the state captured by ``token``."""

    @abc.abstractmethod
    def teardown(self) -> None:
        """Stop/clean up the environment. Must be safe to call more than once."""

    def __enter__(self) -> "Sandbox":
        self.setup()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.teardown()


class SandboxExecutor:
    """Bridges the registry's dangerous-tool gate to a concrete ``Sandbox``.

    The registry refuses to run a ``dangerous`` tool unless a sandbox executor is
    wired in; this is that executor. It implements the two v1 sandbox tools
    (``run_shell`` and ``sandbox_rollback``) against the ``Sandbox``:

    * ``run_shell`` auto-snapshots *before* executing (so the call is always
      revertible) and returns the command result plus the pre-exec snapshot token.
    * ``sandbox_rollback`` restores a given snapshot token.

    Any other dangerous tool raises ``SandboxRequiredError`` - there is no
    in-process fallback, by design.
    """

    def __init__(self, sandbox: Sandbox) -> None:
        self.sandbox = sandbox
        self._n = 0
        # The token snapshotted before the most recent run_shell, so a caller can
        # revert "the last command" without knowing the opaque token string.
        self._last_snapshot: Optional[str] = None

    def __call__(self, tool: Tool, args: Dict[str, Any]) -> Any:
        name = tool.spec.name
        if name == "run_shell":
            command = str(args["command"])
            self._n += 1
            token = self.sandbox.snapshot(f"pre-run_shell-{self._n}")
            self._last_snapshot = token
            result = self.sandbox.exec(command)
            out = result.to_dict()
            # Hand the snapshot token back so the model (or a demo) can revert
            # this exact command with sandbox_rollback.
            out["snapshot"] = token
            return out
        if name == "sandbox_rollback":
            # Explicit token if given; otherwise revert the most recent run_shell.
            token = args.get("snapshot") or self._last_snapshot
            if not token:
                raise SandboxRequiredError(
                    "sandbox_rollback: no snapshot given and no prior run_shell to revert"
                )
            self.sandbox.rollback(str(token))
            return f"rolled back to snapshot {token}"
        raise SandboxRequiredError(
            f"no sandbox handler for dangerous tool {name!r} (v1 sandboxes "
            f"run_shell / sandbox_rollback)"
        )
