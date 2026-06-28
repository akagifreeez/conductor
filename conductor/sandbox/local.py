"""SubprocessSandbox - the offline sandbox double.

Runs commands in a throwaway temp directory; snapshot is a copy of that
directory, rollback restores the copy. It executes *real* commands and really
reverts them, so the snapshot/exec/rollback contract is demonstrable with no
Proxmox node - which is what the offline tests and the demo use.

⚠️ NOT a security boundary. The command runs as the current user with a working
directory set to the box, but nothing stops it touching absolute paths outside
the box (``rm -rf /``, reading secrets, network). Real isolation is
``ProxmoxSandbox``. This class exists to prove the *flow*, not to contain hostile
code - and the README says so plainly.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Optional

from .base import ExecResult, Sandbox


class SubprocessSandbox(Sandbox):
    """A filesystem-snapshot sandbox backed by a temp directory."""

    kind = "subprocess (NOT a security boundary)"

    def __init__(self, *, name: str = "subprocess-1", seed_files: Optional[dict] = None) -> None:
        self.name = name
        self._root: Optional[str] = None     # parent temp dir (box + snapshots)
        self._box: Optional[str] = None      # the working directory commands see
        self._snaps_dir: Optional[str] = None
        self._seed_files = dict(seed_files or {})
        self._n = 0

    # -- lifecycle ------------------------------------------------------

    def setup(self) -> None:
        self._root = tempfile.mkdtemp(prefix="conductor-sbx-")
        self._box = os.path.join(self._root, "box")
        self._snaps_dir = os.path.join(self._root, "snapshots")
        os.makedirs(self._box, exist_ok=True)
        os.makedirs(self._snaps_dir, exist_ok=True)
        # Seed any initial files so a demo has something to destroy/restore.
        for rel, content in self._seed_files.items():
            path = os.path.join(self._box, rel)
            os.makedirs(os.path.dirname(path) or self._box, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)

    def teardown(self) -> None:
        if self._root and os.path.isdir(self._root):
            shutil.rmtree(self._root, ignore_errors=True)
        self._root = self._box = self._snaps_dir = None

    # -- snapshot / exec / rollback ------------------------------------

    def snapshot(self, label: str) -> str:
        self._require_setup()
        self._n += 1
        token = f"{self._n}-{label}"
        dest = os.path.join(self._snaps_dir, token)
        # Full copy of the box = the snapshot. dirs_exist_ok keeps it simple.
        shutil.copytree(self._box, dest, dirs_exist_ok=True)
        return token

    def exec(self, command: str, *, timeout: float = 60.0) -> ExecResult:
        self._require_setup()
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=self._box,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return ExecResult(exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
        except subprocess.TimeoutExpired as e:
            return ExecResult(exit_code=124, stdout=e.stdout or "", stderr=f"timeout after {timeout}s")

    def rollback(self, token: str) -> None:
        self._require_setup()
        snap = os.path.join(self._snaps_dir, token)
        if not os.path.isdir(snap):
            raise ValueError(f"unknown snapshot token: {token!r}")
        # Restore atomically: build the new box in a temp dir FIRST, then swap it
        # in. If the copy fails partway, the live box is left untouched (a
        # rmtree-then-copytree order would destroy the box before knowing the
        # restore can succeed).
        staged = os.path.join(self._root, f"_restore-{token}")  # type: ignore[arg-type]
        shutil.rmtree(staged, ignore_errors=True)
        shutil.copytree(snap, staged)
        shutil.rmtree(self._box, ignore_errors=True)
        os.replace(staged, self._box)

    # -- helpers (used by tests/demo) ----------------------------------

    def list_files(self) -> list:
        """Sorted relative paths currently in the box (for assertions/demos)."""
        self._require_setup()
        out = []
        for dirpath, _dirs, files in os.walk(self._box):
            for f in files:
                out.append(os.path.relpath(os.path.join(dirpath, f), self._box).replace("\\", "/"))
        return sorted(out)

    @property
    def box_dir(self) -> str:
        self._require_setup()
        return self._box  # type: ignore[return-value]

    def _require_setup(self) -> None:
        if not self._box:
            raise RuntimeError("sandbox not set up; call setup() (or use as a context manager)")
