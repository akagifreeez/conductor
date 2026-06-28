"""ProxmoxSSHSandbox - a real LXC sandbox driven entirely over SSH `pct`.

This is the no-API-token path, and the natural fit for **Tailscale SSH**: auth is
handled by the SSH / Tailscale layer, so all this class needs is that
``ssh <user>@<host> 'pct ...'`` runs as root on the Proxmox node. Lifecycle,
snapshot, exec, and rollback are all the node's own ``pct`` subcommands:

* setup   -> ``pct create`` (if a template is given) then ``pct start``
* snapshot-> ``pct snapshot <vmid> <name>``
* exec    -> ``pct exec <vmid> -- sh -lc '<cmd>'``
* rollback-> ``pct rollback <vmid> <name>`` then ``pct start`` (rollback stops the CT)
* teardown-> ``pct stop`` + ``pct destroy`` (only if WE created the CT)

The actual subprocess call is injected (``runner``) so the command-construction
logic is unit-tested offline without a node; the default runner shells out to the
system ``ssh``. Snapshots capture the rootfs disk, so the self-check marker lives
in ``/root`` (persistent), not ``/tmp`` (often tmpfs).
"""
from __future__ import annotations

import re
import shlex
import subprocess
import time
from typing import Callable, List, Optional, Tuple

from .base import ExecResult, Sandbox

# A runner takes argv and returns (returncode, stdout, stderr).
Runner = Callable[[List[str]], Tuple[int, str, str]]

# Snapshot tokens / hostnames that are safe to interpolate into a remote `pct`
# command. Tokens can arrive from a model tool call (sandbox_rollback), so they
# MUST be validated before they reach the node's shell.
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_-]+$")
# SSH destination parts: no leading '-' (option injection), restricted charset.
_SAFE_DEST = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]*$")


def _default_runner(timeout: float) -> Runner:
    def run(argv: List[str]) -> Tuple[int, str, str]:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    return run


class ProxmoxSSHSandbox(Sandbox):
    """An LXC sandbox controlled via ``pct`` over SSH (no Proxmox API token)."""

    kind = "proxmox-lxc via ssh (real OS isolation)"

    def __init__(
        self,
        vmid: int,
        *,
        host: str,
        ssh_user: str = "root",
        ssh_bin: str = "ssh",
        connect_timeout: int = 15,
        command_timeout: float = 120.0,
        template_volume: Optional[str] = None,
        storage: str = "local-lvm",
        rootfs_gb: int = 2,
        memory_mb: int = 512,
        hostname: str = "conductor-selfcheck",
        runner: Optional[Runner] = None,
        name: str = "proxmox-ssh",
    ) -> None:
        self.name = name
        self.vmid = int(vmid)
        # host/ssh_user become part of the ssh argv ("user@host"); reject values
        # that could smuggle an ssh option or shell metacharacter.
        if not _SAFE_DEST.match(host):
            raise ValueError(f"unsafe ssh host: {host!r}")
        if not _SAFE_DEST.match(ssh_user):
            raise ValueError(f"unsafe ssh user: {ssh_user!r}")
        self.host = host
        self.ssh_user = ssh_user
        self.ssh_bin = ssh_bin
        self.connect_timeout = int(connect_timeout)
        self.command_timeout = float(command_timeout)
        self.template_volume = template_volume   # e.g. "local:vztmpl/debian-13-...tar.zst"
        self.storage = storage
        self.rootfs_gb = int(rootfs_gb)
        self.memory_mb = int(memory_mb)
        self.hostname = hostname
        self._runner = runner or _default_runner(self.command_timeout)
        self._snap_n = 0
        self._created = False

    # -- ssh plumbing ---------------------------------------------------

    def _ssh(self, remote_cmd: str) -> ExecResult:
        argv = [
            self.ssh_bin,
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"ConnectTimeout={self.connect_timeout}",
            f"{self.ssh_user}@{self.host}",
            remote_cmd,
        ]
        try:
            code, out, err = self._runner(argv)
        except subprocess.TimeoutExpired:
            return ExecResult(exit_code=124, stdout="", stderr=f"ssh timeout: {remote_cmd}")
        return ExecResult(exit_code=code, stdout=out, stderr=err)

    def _pct(self, pct_args: str) -> ExecResult:
        return self._ssh(f"pct {pct_args}")

    def _status(self) -> str:
        r = self._ssh(f"pct status {self.vmid}")
        # "status: running" -> "running"
        return r.stdout.strip().split()[-1] if r.ok and r.stdout.strip() else ""

    def _await_status(self, want: str, *, tries: int = 30, delay: float = 1.0) -> None:
        for _ in range(tries):
            if self._status() == want:
                return
            time.sleep(delay)
        raise RuntimeError(f"CT {self.vmid} did not reach status {want!r} in {tries * delay:.0f}s")

    # -- Sandbox interface ----------------------------------------------

    def setup(self) -> None:
        if self.template_volume:
            r = self._pct(
                f"create {self.vmid} {shlex.quote(self.template_volume)} "
                f"--hostname {shlex.quote(self.hostname)} --memory {self.memory_mb} "
                f"--rootfs {shlex.quote(self.storage)}:{self.rootfs_gb} "
                f"--unprivileged 1 --cores 1"
            )
            if not r.ok:
                raise RuntimeError(f"pct create failed: {r.stderr or r.stdout}")
            self._created = True
        r = self._pct(f"start {self.vmid}")
        if not r.ok and "already running" not in (r.stderr + r.stdout).lower():
            raise RuntimeError(f"pct start failed: {r.stderr or r.stdout}")
        self._await_status("running")

    def snapshot(self, label: str) -> str:
        self._snap_n += 1
        clean = "".join(c if c.isalnum() else "_" for c in label)[:20]
        snapname = f"cdr{self._snap_n}_{clean}"
        r = self._pct(f"snapshot {self.vmid} {snapname}")
        if not r.ok:
            raise RuntimeError(f"pct snapshot failed: {r.stderr or r.stdout}")
        return snapname

    def exec(self, command: str, *, timeout: float = 60.0) -> ExecResult:
        # `pct exec <vmid> -- sh -lc '<command>'`; shlex.quote protects the inner
        # command through the remote shell that ssh runs the whole string in.
        remote = f"pct exec {self.vmid} -- sh -lc {shlex.quote(command)}"
        return self._ssh(remote)

    def rollback(self, token: str) -> None:
        # token can originate from a model tool call (sandbox_rollback) - validate
        # it before it touches the node's shell, so it can't inject a `pct destroy`
        # or arbitrary command. (snapshot() only ever emits matching tokens.)
        if not _SAFE_TOKEN.match(token or ""):
            raise ValueError(f"unsafe snapshot token: {token!r}")
        # pct rollback stops the CT and restores the rootfs; bring it back up so
        # the next exec/probe runs against the restored state.
        r = self._pct(f"rollback {self.vmid} {token}")
        if not r.ok:
            raise RuntimeError(f"pct rollback failed: {r.stderr or r.stdout}")
        start = self._pct(f"start {self.vmid}")
        if not start.ok and "already running" not in (start.stderr + start.stdout).lower():
            raise RuntimeError(f"pct start after rollback failed: {start.stderr or start.stdout}")
        self._await_status("running")

    def teardown(self) -> None:
        # Only ever destroy a CT WE created; a pre-existing CT is left exactly as
        # found (not even stopped). _created is set True only after a successful
        # `pct create`, so a production CT can never reach the destroy path.
        if not self._created:
            return
        self._pct(f"stop {self.vmid}")
        for _ in range(20):
            if self._status() in ("stopped", ""):
                break
            time.sleep(1.0)
        r = self._pct(f"destroy {self.vmid} --purge 1")
        if not r.ok and self._status() != "":
            # Don't silently drop a leaked throwaway CT under a PASS verdict: keep
            # _created True (so a retry is possible) and surface the failure.
            raise RuntimeError(
                f"pct destroy {self.vmid} failed (CT may be leaked): {r.stderr or r.stdout}"
            )
        self._created = False
