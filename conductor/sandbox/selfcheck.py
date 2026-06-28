"""A sandbox-agnostic self-check: prove snapshot/exec/rollback on ANY Sandbox.

The same routine runs against the offline ``SubprocessSandbox`` (so the logic is
unit-tested cross-platform) and against a real ``ProxmoxSandbox`` on your homelab
(so you can verify OS isolation with one command - see ``conductor proxmox-check``).

The cycle: seed a marker -> snapshot -> run a destructive command -> confirm the
marker is gone -> rollback -> confirm the marker is back. The shell commands are
injected (POSIX defaults for a Linux LXC; the offline test passes portable
Python one-liners), so the *verification logic* is what's shared and tested, not
a particular command syntax.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .base import Sandbox

# Marker written into the sandbox and a token printed when it's absent.
DEFAULT_MARKER = "conductor-selfcheck-OK"
DEFAULT_ABSENT = "__ABSENT__"
# Persistent, snapshot-backed path. NOT /tmp: on many LXC templates /tmp is a
# tmpfs (RAM), which a Proxmox disk snapshot does not capture - and rollback
# reboots the CT, wiping tmpfs, so the marker would be "lost" and the check would
# spuriously FAIL on a perfectly working snapshot/rollback. /root is on the rootfs.
_FILE = "/root/conductor_selfcheck"


def posix_commands(marker: str = DEFAULT_MARKER, absent: str = DEFAULT_ABSENT, path: str = _FILE):
    """POSIX shell commands for a Linux LXC (used by proxmox-check)."""
    return {
        "seed_cmd": f"echo {marker} > {path} && cat {path}",
        "destroy_cmd": f"rm -f {path}",
        "probe_cmd": f"cat {path} 2>/dev/null || echo {absent}",
        "marker": marker,
        "absent": absent,
    }


@dataclass
class SelfCheckReport:
    steps: List[Tuple[str, bool, str]] = field(default_factory=list)
    passed: bool = False
    error: Optional[str] = None

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.steps.append((name, ok, detail))

    def format(self) -> str:
        lines = []
        for name, ok, detail in self.steps:
            mark = "PASS" if ok else "FAIL"
            lines.append(f"  [{mark}] {name}" + (f" - {detail}" if detail else ""))
        if self.error:
            lines.append(f"  [FAIL] error: {self.error}")
        lines.append(f"\n{'PASS' if self.passed else 'FAIL'}: sandbox snapshot/rollback self-check")
        return "\n".join(lines)


def sandbox_selfcheck(
    sandbox: Sandbox,
    *,
    seed_cmd: str,
    destroy_cmd: str,
    probe_cmd: str,
    marker: str = DEFAULT_MARKER,
    absent: str = DEFAULT_ABSENT,
) -> SelfCheckReport:
    """Run the seed->snapshot->destroy->probe->rollback->probe cycle on ``sandbox``.

    Returns a report whose ``passed`` is True only if: the marker seeded, the
    destructive command actually removed it (contained), and rollback restored
    it. Always tears the sandbox down.
    """
    report = SelfCheckReport()
    try:
        sandbox.setup()
        report.add("setup", True, f"{sandbox.kind}")

        seeded = sandbox.exec(seed_cmd)
        seed_ok = marker in seeded.stdout
        report.add("seed marker", seed_ok, _trim(seeded.stdout))

        token = sandbox.snapshot("selfcheck")
        # "call returned without raising" - the meaningful verification is the
        # marker-gone and marker-restored probes below, not this row.
        report.add("snapshot (call ok)", True, f"token={token}")

        destroyed = sandbox.exec(destroy_cmd)
        report.add("destructive command", destroyed.ok, _trim(destroyed.stderr or destroyed.stdout))

        after = sandbox.exec(probe_cmd)
        gone_ok = (marker not in after.stdout) and (absent in after.stdout or not after.stdout.strip())
        report.add("marker gone after destroy (contained)", gone_ok, _trim(after.stdout))

        sandbox.rollback(token)
        report.add("rollback (call ok)", True, f"token={token}")

        restored = sandbox.exec(probe_cmd)
        restore_ok = marker in restored.stdout
        report.add("marker restored after rollback", restore_ok, _trim(restored.stdout))

        report.passed = seed_ok and gone_ok and restore_ok
    except Exception as e:  # noqa: BLE001 - surface as a failed report, not a crash
        report.error = f"{type(e).__name__}: {e}"
        report.passed = False
    finally:
        try:
            sandbox.teardown()
        except Exception as e:  # noqa: BLE001 - surface cleanup failure, don't hide it
            # A teardown failure (e.g. a leaked throwaway CT) must be visible even
            # under a PASS verdict, rather than silently swallowed.
            report.add("teardown", False, f"{type(e).__name__}: {e}")
    return report


def _trim(s: str, n: int = 80) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[:n] + "..."
