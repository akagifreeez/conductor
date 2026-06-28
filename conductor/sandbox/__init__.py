"""Sandboxes where dangerous tools execute (with snapshot/rollback)."""
from .base import ExecResult, Sandbox, SandboxExecutor
from .local import SubprocessSandbox
from .docker import DockerSandbox
from .proxmox import ProxmoxSandbox
from .proxmox_ssh import ProxmoxSSHSandbox
from .selfcheck import SelfCheckReport, posix_commands, sandbox_selfcheck

__all__ = [
    "ExecResult",
    "Sandbox",
    "SandboxExecutor",
    "SubprocessSandbox",
    "DockerSandbox",
    "ProxmoxSandbox",
    "ProxmoxSSHSandbox",
    "SelfCheckReport",
    "sandbox_selfcheck",
    "posix_commands",
]
