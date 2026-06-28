"""DockerSandbox - OS-level isolation in a Docker container (no Proxmox needed).

This is the *universal* real-isolation backend: Docker runs on almost any dev
machine, so the "dangerous tools run isolated, with snapshot/rollback" guarantee
no longer needs a homelab. It plugs into the exact same ``Sandbox`` interface as
the Proxmox backends, so the orchestrator, the gate, the trace, and
``sandbox_selfcheck`` are all unchanged.

Mapping onto Docker:

* setup    -> ``docker run -d --name <c> <image> sh -c 'tail -f /dev/null'``
* snapshot -> ``docker commit <c> <tag>``  (captures the container filesystem;
  the returned image tag IS the token)
* exec     -> ``docker exec <c> sh -c '<cmd>'``
* rollback -> ``docker rm -f <c>`` then ``docker run`` from the committed image
  (restores the filesystem state; same disk-snapshot model as an LXC)
* teardown -> ``docker rm -f`` the container + ``docker rmi`` our snapshot images

Honesty (no overstatement): a Docker container is
**OS-level isolation that shares the host kernel** (namespaces/cgroups) - the same
tier as a Proxmox LXC, *not* a full VM. It contains a misbehaving command and is
revertible, but it is not a hardened boundary against a kernel exploit; for that,
use a microVM/KVM. This is stated so the isolation claim isn't overstated.

The subprocess call is injected (``runner``) so the whole cycle is unit-tested
offline against a simulated daemon; the default runner shells out to ``docker``.
"""
from __future__ import annotations

import re
import subprocess
import time
import uuid
from typing import Callable, List, Optional, Tuple

from .base import ExecResult, Sandbox

Runner = Callable[[List[str]], Tuple[int, str, str]]

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")          # docker container/image names
_SAFE_IMAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./:-]*$")       # repo[:tag]


def _default_runner(timeout: float) -> Runner:
    def run(argv: List[str]) -> Tuple[int, str, str]:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    return run


class DockerSandbox(Sandbox):
    """A ``Sandbox`` backed by a Docker container (snapshot via ``docker commit``)."""

    kind = "docker container (OS-level isolation, shared kernel - LXC tier, not a VM)"

    def __init__(
        self,
        *,
        image: str = "alpine:3",
        container: Optional[str] = None,
        docker_bin: str = "docker",
        keepalive: str = "tail -f /dev/null",
        command_timeout: float = 120.0,
        runner: Optional[Runner] = None,
        name: str = "docker",
        container_suffix: str = "test",
    ) -> None:
        self.name = name
        # NOTE: `image` is trusted *operator* input (it selects which image to pull
        # and run). It is shape-validated only. The model-facing path is the
        # rollback token, which is allow-listed against snapshots WE created.
        if not _SAFE_IMAGE.match(image):
            raise ValueError(f"unsafe docker image: {image!r}")
        self.image = image
        # Per-instance id so the default container name (and thus every snapshot
        # image tag derived from it) is unique on the daemon. Without this, two
        # sandboxes sharing the constant default name would mint colliding commit
        # tags, orphaning images and risking a wrong-layer rollback.
        self._uid = uuid.uuid4().hex[:8]
        self.container = container or f"conductor-sbx-{container_suffix}-{self._uid}"
        if not _SAFE_NAME.match(self.container):
            raise ValueError(f"unsafe container name: {self.container!r}")
        self.docker_bin = docker_bin
        self.keepalive = keepalive
        self.command_timeout = float(command_timeout)
        self._runner = runner or _default_runner(self.command_timeout)
        self._snap_n = 0
        self._snapshots: List[str] = []   # image tags we created (for rollback allow-list + cleanup)
        self._created = False

    # -- docker plumbing -----------------------------------------------

    def _docker(self, args: List[str]) -> ExecResult:
        try:
            code, out, err = self._runner([self.docker_bin, *args])
        except subprocess.TimeoutExpired:
            return ExecResult(exit_code=124, stdout="", stderr=f"docker timeout: {' '.join(args)}")
        return ExecResult(exit_code=code, stdout=out, stderr=err)

    # -- Sandbox interface ---------------------------------------------

    def setup(self) -> None:
        # Fresh start: clear any prior snapshot tracking for this instance.
        self._snapshots = []
        self._snap_n = 0
        # Remove any stale container with our name, then start fresh.
        self._docker(["rm", "-f", self.container])
        r = self._docker(
            ["run", "-d", "--name", self.container, self.image, "sh", "-c", self.keepalive]
        )
        if not r.ok:
            raise RuntimeError(f"docker run failed: {r.stderr or r.stdout}")
        self._created = True
        self._await_running()

    def _await_running(self, *, tries: int = 20, delay: float = 0.5) -> None:
        # `docker run -d` returns once the container is created; confirm it is
        # actually running (not instantly-exited) before we exec against it.
        for _ in range(tries):
            r = self._docker(["inspect", "-f", "{{.State.Running}}", self.container])
            if r.ok and r.stdout.strip() == "true":
                return
            time.sleep(delay)
        raise RuntimeError(f"container {self.container} did not reach a running state")

    def snapshot(self, label: str) -> str:
        self._snap_n += 1
        clean = "".join(c if c.isalnum() else "_" for c in label).lower()[:20]
        tag = f"conductor-snap-{self.container}:{self._snap_n}_{clean}"
        r = self._docker(["commit", self.container, tag])
        if not r.ok:
            raise RuntimeError(f"docker commit failed: {r.stderr or r.stdout}")
        self._snapshots.append(tag)
        return tag

    def exec(self, command: str, *, timeout: float = 60.0) -> ExecResult:
        # command is one argv element handed to `sh -c` INSIDE the container; with
        # an argv list (no shell=True) there is no host-shell injection.
        return self._docker(["exec", self.container, "sh", "-c", command])

    def rollback(self, token: str) -> None:
        # Only roll back to a snapshot WE created - a token can come from a model
        # tool call, and `docker run <token>` would otherwise pull/run an arbitrary
        # image. Restrict to our own commit tags.
        if token not in self._snapshots:
            raise ValueError(f"unknown snapshot token (not created by this sandbox): {token!r}")
        self._docker(["rm", "-f", self.container])
        r = self._docker(
            ["run", "-d", "--name", self.container, token, "sh", "-c", self.keepalive]
        )
        if not r.ok:
            raise RuntimeError(f"docker run (rollback) failed: {r.stderr or r.stdout}")
        self._await_running()

    def teardown(self) -> None:
        # Only clean up resources WE created.
        if not self._created:
            return
        self._docker(["rm", "-f", self.container])
        leaked = []
        for tag in self._snapshots:
            r = self._docker(["rmi", "-f", tag])
            if not r.ok:
                leaked.append(tag)
        self._snapshots = []
        self._created = False
        if leaked:
            raise RuntimeError(f"failed to remove snapshot image(s) (may be leaked): {leaked}")
