"""ProxmoxSSHSandbox: full self-check cycle against a SIMULATED node (no network).

A fake ``runner`` stands in for ``ssh ... pct ...``, maintaining a tiny in-memory
container filesystem + snapshots, so the command construction and the
snapshot/destroy/rollback semantics are proven offline. This is the same
``sandbox_selfcheck`` routine the live ``conductor proxmox-check`` runs.
"""
from __future__ import annotations

import shlex

from conductor.sandbox import ProxmoxSSHSandbox, posix_commands, sandbox_selfcheck


class FakeNode:
    """Simulates a Proxmox node's `pct` behavior over the injected runner."""

    def __init__(self, *, fail_destroy=False):
        self.fs = {}                 # path -> content (the CT rootfs we care about)
        self.snaps = {}              # name -> fs copy
        self.status = "absent"
        self.created = False
        self.destroyed = False
        self.fail_destroy = fail_destroy
        self.calls = []

    def __call__(self, argv):
        remote = argv[-1]            # ssh runs the last arg via the remote shell
        self.calls.append(remote)
        toks = shlex.split(remote)
        assert toks[0] == "pct", remote
        sub = toks[1]
        if sub == "create":
            self.created = True
            self.status = "stopped"
            return 0, f"created CT {toks[2]}\n", ""
        if sub == "start":
            self.status = "running"
            return 0, "", ""
        if sub == "stop":
            self.status = "stopped"
            return 0, "", ""
        if sub == "status":
            return 0, f"status: {self.status}\n", ""
        if sub == "snapshot":
            self.snaps[toks[3]] = dict(self.fs)
            return 0, "", ""
        if sub == "rollback":
            self.fs = dict(self.snaps[toks[3]])
            self.status = "stopped"
            return 0, "", ""
        if sub == "destroy":
            if self.fail_destroy:
                return 1, "", "CT is busy"   # status stays as-is (e.g. running)
            self.destroyed = True
            self.status = "absent"
            return 0, "", ""
        if sub == "exec":
            return self._exec(toks)
        return 1, "", f"unknown pct subcommand: {sub}"

    def _exec(self, toks):
        # toks = ["pct","exec","<vmid>","--","sh","-lc","<inner>"]
        inner = toks[toks.index("-lc") + 1]
        if "&&" in inner:                       # seed: echo M > P && cat P
            left, _right = inner.split("&&", 1)
            parts = left.split(">")
            marker = parts[0].replace("echo", "", 1).strip()
            path = parts[1].strip()
            self.fs[path] = marker
            return 0, self.fs[path] + "\n", ""
        if inner.startswith("rm -f"):           # destroy
            path = inner.split()[-1]
            self.fs.pop(path, None)
            return 0, "", ""
        if inner.startswith("cat") and "||" in inner:   # probe: cat P ... || echo A
            path = shlex.split(inner)[1]
            absent = inner.split("echo", 1)[1].strip()
            if path in self.fs:
                return 0, self.fs[path] + "\n", ""
            return 0, absent + "\n", ""
        return 1, "", f"unhandled inner: {inner}"


def test_proxmox_ssh_selfcheck_full_cycle_simulated():
    node = FakeNode()
    sb = ProxmoxSSHSandbox(
        vmid=101, host="100.100.1.1",
        template_volume="local:vztmpl/debian-13-standard_13.1-2_amd64.tar.zst",
        runner=node,
    )
    report = sandbox_selfcheck(sb, **posix_commands())

    assert report.passed is True, report.format()
    # the CT we created was destroyed (no leak)
    assert node.created and node.destroyed
    # the marker really left the fs after destroy and came back after rollback
    names = [s[0] for s in report.steps]
    assert "marker gone after destroy (contained)" in names
    assert "marker restored after rollback" in names
    assert all(ok for _n, ok, _d in report.steps)


def test_proxmox_ssh_builds_expected_pct_commands():
    node = FakeNode()
    sb = ProxmoxSSHSandbox(vmid=101, host="h", template_volume="local:vztmpl/x.tar.zst", runner=node)
    sandbox_selfcheck(sb, **posix_commands())
    joined = "\n".join(node.calls)
    assert "pct create 101 local:vztmpl/x.tar.zst" in joined
    assert "pct snapshot 101 " in joined
    assert "pct exec 101 -- sh -lc " in joined
    assert "pct rollback 101 " in joined
    assert "pct destroy 101 --purge 1" in joined


def test_proxmox_ssh_does_not_create_or_destroy_preexisting_ct():
    # No template_volume -> use an existing CT: never create, never destroy.
    node = FakeNode()
    node.status = "running"  # pretend CT already exists & runs
    sb = ProxmoxSSHSandbox(vmid=110, host="h", runner=node)  # 110 = "production"
    sandbox_selfcheck(sb, **posix_commands())
    assert node.created is False
    assert node.destroyed is False


# --- security/robustness fixes from the safety review --------------------

import pytest


def test_rollback_rejects_unsafe_token_injection():
    # A model-supplied token must not be able to inject a `pct destroy`/shell cmd.
    sb = ProxmoxSSHSandbox(vmid=101, host="h", runner=FakeNode())
    with pytest.raises(ValueError):
        sb.rollback("1; pct destroy 110")
    with pytest.raises(ValueError):
        sb.rollback("$(rm -rf /)")


def test_unsafe_ssh_host_or_user_rejected():
    with pytest.raises(ValueError):
        ProxmoxSSHSandbox(vmid=1, host="-oProxyCommand=evil", runner=FakeNode())
    with pytest.raises(ValueError):
        ProxmoxSSHSandbox(vmid=1, host="h", ssh_user="-x", runner=FakeNode())
    # a normal Tailscale IP + root must be accepted
    ProxmoxSSHSandbox(vmid=1, host="100.100.1.1", ssh_user="root", runner=FakeNode())


def test_teardown_destroy_failure_is_surfaced_not_swallowed():
    node = FakeNode(fail_destroy=True)
    sb = ProxmoxSSHSandbox(
        vmid=101, host="h", template_volume="local:vztmpl/x.tar.zst", runner=node,
    )
    report = sandbox_selfcheck(sb, **posix_commands())
    # the snapshot/rollback contract still PASSes...
    assert report.passed is True
    # ...but the leaked-CT teardown failure is visible as a FAIL row, not hidden.
    assert any(name == "teardown" and not ok for name, ok, _ in report.steps)
