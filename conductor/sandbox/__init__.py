"""Sandboxes where dangerous tools execute (with snapshot/rollback)."""
from .base import ExecResult, Sandbox, SandboxExecutor
from .local import SubprocessSandbox
from .proxmox import ProxmoxSandbox

__all__ = [
    "ExecResult",
    "Sandbox",
    "SandboxExecutor",
    "SubprocessSandbox",
    "ProxmoxSandbox",
]
