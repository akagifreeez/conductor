"""Sandboxes where dangerous tools execute (with snapshot/rollback)."""
from .base import ExecResult, Sandbox, SandboxExecutor
from .local import SubprocessSandbox
from .proxmox import ProxmoxSandbox
from .selfcheck import SelfCheckReport, posix_commands, sandbox_selfcheck

__all__ = [
    "ExecResult",
    "Sandbox",
    "SandboxExecutor",
    "SubprocessSandbox",
    "ProxmoxSandbox",
    "SelfCheckReport",
    "sandbox_selfcheck",
    "posix_commands",
]
