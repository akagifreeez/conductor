"""DockerSandbox: full self-check cycle against a SIMULATED daemon (no Docker).

A fake ``runner`` stands in for the ``docker`` CLI, maintaining in-memory
containers (each with a tiny filesystem) and committed images, so the
run/commit/exec/rollback/rmi semantics are proven offline - the same
``sandbox_selfcheck`` routine the live `sandbox-check --backend docker` runs.
"""
from __future__ import annotations

import shlex

import pytest

from conductor.sandbox import DockerSandbox, posix_commands, sandbox_selfcheck


class FakeDocker:
    def __init__(self, *, fail_rmi=False):
        self.containers = {}     # name -> fs dict
        self.images = {}         # tag -> fs dict (committed snapshots)
        self.fail_rmi = fail_rmi
        self.calls = []

    def __call__(self, argv):
        assert argv[0] == "docker"
        self.calls.append(argv)
        sub = argv[1]
        if sub == "rm":          # rm -f <name>
            self.containers.pop(argv[-1], None)
            return 0, "", ""
        if sub == "run":         # run -d --name <c> <image> sh -c <keepalive>
            name = argv[argv.index("--name") + 1]
            image = argv[argv.index("--name") + 2]
            base = dict(self.images.get(image, {}))   # restore from committed image, else empty
            self.containers[name] = base
            return 0, f"{name}\n", ""
        if sub == "commit":      # commit <c> <tag>
            c, tag = argv[2], argv[3]
            if c not in self.containers:
                return 1, "", "no such container"
            self.images[tag] = dict(self.containers[c])
            return 0, "", ""
        if sub == "inspect":     # inspect -f {{.State.Running}} <name>
            name = argv[-1]
            return (0, "true\n", "") if name in self.containers else (0, "false\n", "")
        if sub == "exec":        # exec <c> sh -c <cmd>
            c = argv[2]
            cmd = argv[argv.index("-c") + 1]
            return self._exec(self.containers.setdefault(c, {}), cmd)
        if sub == "rmi":         # rmi -f <tag>
            if self.fail_rmi:
                return 1, "", "image is in use"
            self.images.pop(argv[-1], None)
            return 0, "", ""
        return 1, "", f"unknown docker subcommand: {sub}"

    @staticmethod
    def _exec(fs, cmd):
        if "&&" in cmd:                       # seed: echo M > P && cat P
            left, _ = cmd.split("&&", 1)
            parts = left.split(">")
            marker = parts[0].replace("echo", "", 1).strip()
            path = parts[1].strip()
            fs[path] = marker
            return 0, fs[path] + "\n", ""
        if cmd.startswith("rm -f"):           # destroy
            fs.pop(cmd.split()[-1], None)
            return 0, "", ""
        if cmd.startswith("cat") and "||" in cmd:    # probe
            path = shlex.split(cmd)[1]
            absent = cmd.split("echo", 1)[1].strip()
            return (0, fs[path] + "\n", "") if path in fs else (0, absent + "\n", "")
        return 1, "", f"unhandled: {cmd}"


def test_docker_selfcheck_full_cycle_simulated():
    fake = FakeDocker()
    sb = DockerSandbox(image="alpine:3", runner=fake, container_suffix="t1")
    report = sandbox_selfcheck(sb, **posix_commands())
    assert report.passed is True, report.format()
    assert all(ok for _n, ok, _d in report.steps)
    # container removed and snapshot image cleaned up (no leak)
    assert sb.container not in fake.containers
    assert fake.images == {}


def test_docker_builds_expected_commands():
    fake = FakeDocker()
    sb = DockerSandbox(image="alpine:3", runner=fake, container_suffix="t2")
    sandbox_selfcheck(sb, **posix_commands())
    flat = [" ".join(c) for c in fake.calls]
    c = sb.container  # uid-suffixed, so reference it rather than hardcoding
    assert any(s.startswith(f"docker run -d --name {c} alpine:3 sh -c") for s in flat)
    assert any(s.startswith(f"docker commit {c} conductor-snap-{c}:") for s in flat)
    assert any(s.startswith(f"docker exec {c} sh -c") for s in flat)
    assert any(s.startswith(f"docker rmi -f conductor-snap-{c}:") for s in flat)
    # the per-instance uid makes the container name (and thus snapshot tags) unique
    assert sb._uid in c


def test_docker_rollback_rejects_foreign_image_token():
    # A model-supplied token must not let `docker run <token>` pull/run any image.
    sb = DockerSandbox(image="alpine:3", runner=FakeDocker(), container_suffix="t3")
    with pytest.raises(ValueError):
        sb.rollback("evil/registry:latest")


def test_docker_unsafe_image_or_container_rejected():
    with pytest.raises(ValueError):
        DockerSandbox(image="-x; rm -rf /", runner=FakeDocker())
    with pytest.raises(ValueError):
        DockerSandbox(image="alpine:3", container="bad name", runner=FakeDocker())


def test_docker_teardown_leaked_image_is_surfaced():
    fake = FakeDocker(fail_rmi=True)
    sb = DockerSandbox(image="alpine:3", runner=fake, container_suffix="t4")
    report = sandbox_selfcheck(sb, **posix_commands())
    assert report.passed is True                      # cycle still verified
    assert any(n == "teardown" and not ok for n, ok, _ in report.steps)  # leak surfaced
