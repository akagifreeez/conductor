"""ProxmoxSandbox - real OS-level isolation in a Proxmox LXC.

This is the actual differentiator: dangerous tools execute inside a real
Linux container on a Proxmox VE node, and snapshot/rollback use Proxmox's own
snapshot mechanism, so a destructive command is genuinely contained and
revertible with the host untouched.

The lifecycle calls mirror the author's **proxmoxbot** (proxmoxer shapes):
``client.nodes(node).lxc(vmid).status.start.post()``,
``...lxc(vmid).snapshot.post(snapname=...)``,
``...lxc(vmid).snapshot(name).rollback.post()``. Command execution is the one
thing Proxmox has no clean REST call for, so it is done over SSH to the node:
``pct exec <vmid> -- sh -lc '<command>'`` (the standard way to run a command in
an LXC).

Honesty (important): this class is written against the real API/SSH but is **not
exercised by the offline test suite** (it needs a live Proxmox node + an
existing container/template). It is the path you run on your own homelab. The
optional deps (``proxmoxer``, ``paramiko``) are imported lazily so nothing else
in Conductor depends on them. Verify against a real node before relying on it.
"""
from __future__ import annotations

import os
import shlex
import time
from typing import Any, Optional

from .base import ExecResult, Sandbox


class ProxmoxSandbox(Sandbox):
    """A ``Sandbox`` backed by a Proxmox LXC container.

    Connection params fall back to environment variables so secrets stay out of
    code: ``PROXMOX_HOST``, ``PROXMOX_USER``, ``PROXMOX_TOKEN_NAME``,
    ``PROXMOX_TOKEN_VALUE`` (API token auth, same as proxmoxbot), plus
    ``PROXMOX_NODE`` and ``PROXMOX_SSH_*`` for the exec channel.

    ``vmid`` is the LXC to use. If ``template_vmid`` is given, ``setup`` clones it
    to a fresh container first (and ``teardown`` destroys the clone); otherwise it
    just starts the existing ``vmid`` (and only stops it on teardown).
    """

    kind = "proxmox-lxc (real OS isolation)"

    def __init__(
        self,
        vmid: int,
        *,
        node: Optional[str] = None,
        host: Optional[str] = None,
        user: Optional[str] = None,
        token_name: Optional[str] = None,
        token_value: Optional[str] = None,
        verify_ssl: bool = False,
        template_vmid: Optional[int] = None,
        ssh_host: Optional[str] = None,
        ssh_user: str = "root",
        ssh_key_path: Optional[str] = None,
        name: str = "proxmox-lxc",
    ) -> None:
        self.name = name
        self.vmid = int(vmid)
        self.node = node or os.environ.get("PROXMOX_NODE", "")
        self._host = host or os.environ.get("PROXMOX_HOST", "")
        self._user = user or os.environ.get("PROXMOX_USER", "")
        self._token_name = token_name or os.environ.get("PROXMOX_TOKEN_NAME", "")
        self._token_value = token_value or os.environ.get("PROXMOX_TOKEN_VALUE", "")
        self._verify_ssl = verify_ssl
        self.template_vmid = template_vmid
        self._ssh_host = ssh_host or os.environ.get("PROXMOX_SSH_HOST") or self._host
        self._ssh_user = os.environ.get("PROXMOX_SSH_USER", ssh_user)
        self._ssh_key_path = ssh_key_path or os.environ.get("PROXMOX_SSH_KEY_PATH")
        self._client: Any = None
        self._cloned = False
        self._snap_n = 0  # makes snapshot names unique regardless of label

    # -- proxmox client (lazy) -----------------------------------------

    def _connect(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from proxmoxer import ProxmoxAPI  # lazy optional dep
        except ImportError as e:  # pragma: no cover - env dependent
            raise RuntimeError(
                "ProxmoxSandbox needs 'proxmoxer'; install conductor-cp[proxmox]"
            ) from e
        if not (self._host and self._user and self._token_name and self._token_value):
            raise RuntimeError(
                "Proxmox connection not configured (set PROXMOX_HOST/USER/"
                "TOKEN_NAME/TOKEN_VALUE)"
            )
        self._client = ProxmoxAPI(
            self._host,
            user=self._user,
            token_name=self._token_name,
            token_value=self._token_value,
            verify_ssl=self._verify_ssl,
        )
        return self._client

    def _ct(self) -> Any:
        """The proxmoxer resource for this LXC: nodes(node).lxc(vmid)."""
        if not self.node:
            raise RuntimeError("PROXMOX_NODE not set")
        return self._connect().nodes(self.node).lxc(self.vmid)

    # -- lifecycle ------------------------------------------------------

    def setup(self) -> None:
        client = self._connect()
        if self.template_vmid is not None:
            # Clone the template to our vmid, then start it. Mirrors proxmoxbot's
            # nodes(node).<type>(id).clone.post shape.
            client.nodes(self.node).lxc(self.template_vmid).clone.post(
                newid=self.vmid, hostname=f"conductor-{self.vmid}"
            )
            self._cloned = True
            self._await_status("stopped")
        self._ct().status.start.post()
        self._await_status("running")

    def teardown(self) -> None:
        # Idempotent: safe to call more than once. After a cloned teardown we drop
        # the client/cloned flags so a second call is a fast no-op (and we never
        # re-poll a deleted CT).
        if self._client is None:
            return
        try:
            self._ct().status.stop.post()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass
        if self._cloned:
            try:
                self._await_status("stopped", required=False)
                self._ct().delete()
            except Exception:  # noqa: BLE001
                pass
            self._cloned = False
        self._client = None

    # -- snapshot / exec / rollback ------------------------------------

    def snapshot(self, label: str) -> str:
        # Proxmox snapshot names must be alnum-ish AND unique; a monotonic counter
        # guarantees uniqueness even if two labels sanitize to the same string.
        self._snap_n += 1
        clean = "".join(c if c.isalnum() else "_" for c in label)[:30]
        snapname = f"cdr{self._snap_n}_{clean}"
        self._ct().snapshot.post(snapname=snapname)
        return snapname

    def rollback(self, token: str) -> None:
        self._ct().snapshot(token).rollback.post()
        # rollback stops the CT; bring it back up.
        self._await_status("stopped")
        self._ct().status.start.post()
        self._await_status("running")

    def exec(self, command: str, *, timeout: float = 60.0) -> ExecResult:
        """Run a command in the LXC via SSH ``pct exec`` to the node."""
        return self._ssh_pct_exec(command, timeout=timeout)

    # -- ssh exec channel ----------------------------------------------

    def _ssh_pct_exec(self, command: str, *, timeout: float) -> ExecResult:
        try:
            import paramiko  # lazy optional dep
        except ImportError as e:  # pragma: no cover - env dependent
            raise RuntimeError(
                "ProxmoxSandbox.exec needs 'paramiko'; install conductor-cp[proxmox]"
            ) from e
        if not self._ssh_host:
            raise RuntimeError("PROXMOX_SSH_HOST not set")
        cli = paramiko.SSHClient()
        cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        cli.connect(
            self._ssh_host,
            username=self._ssh_user,
            key_filename=self._ssh_key_path,
            timeout=min(timeout, 30.0),
        )
        try:
            # `pct exec <vmid> -- sh -lc '<command>'` runs the command inside the CT.
            inner = f"sh -lc {shlex.quote(command)}"
            full = f"pct exec {self.vmid} -- {inner}"
            _stdin, stdout, stderr = cli.exec_command(full, timeout=timeout)
            out = stdout.read().decode("utf-8", "replace")
            err = stderr.read().decode("utf-8", "replace")
            code = stdout.channel.recv_exit_status()
            return ExecResult(exit_code=code, stdout=out, stderr=err)
        finally:
            cli.close()

    # -- helpers --------------------------------------------------------

    def _await_status(
        self, want: str, *, tries: int = 30, delay: float = 1.0, required: bool = True
    ) -> None:
        """Poll until the CT reaches ``want``. Raises on timeout unless
        ``required=False`` (used by best-effort teardown).

        Failing loud is deliberate: a snapshot/rollback safety layer must not
        return as if the CT is up when it isn't - a silent timeout would let the
        next exec hit a stopped container with a confusing error, or (after
        rollback) report a restored-and-live box that is actually stopped.
        """
        for _ in range(tries):
            try:
                cur = self._ct().status.current.get().get("status")
            except Exception:  # noqa: BLE001 - transient during clone/start
                cur = None
            if cur == want:
                return
            time.sleep(delay)
        if required:
            raise RuntimeError(
                f"CT {self.vmid} did not reach status {want!r} within {tries * delay:.0f}s"
            )
