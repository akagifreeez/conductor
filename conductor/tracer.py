"""The run trace - every LLM call and tool call, streamed to JSONL.

One line per event, flushed as it happens, so a crash mid-run still leaves an
inspectable partial trace (same durability choice as token-router's Ledger).
The trace is the substrate for two of Conductor's three differentiators:

* **observability** - the full, ordered I/O of a run is on disk, not lost in
  stdout.
* **deterministic replay** (v1) - a recorded trace can be replayed by feeding
  the captured tool outputs back into the loop. "Deterministic" means the
  recorded *I/O* is reproduced, NOT that the LLM is re-run bit-for-bit.

Events are intentionally provider-agnostic: they record the neutral
``Message``/``ToolCall`` shapes, never a provider's wire format, so a trace reads
the same whoever produced it.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from .backends.base import AssistantTurn, ToolCall, ToolResult, ToolSpec


class Tracer:
    """Append-only JSONL writer for run events.

    ``run_id`` tags every line so multiple runs can share a directory and still
    be separated later. Pass ``path`` to control the file; otherwise it is
    ``traces/run-<run_id>.jsonl``.
    """

    def __init__(self, run_id: str, *, path: Optional[str] = None, trace_dir: str = "traces") -> None:
        self.run_id = run_id
        self.path = path or os.path.join(trace_dir, f"run-{run_id}.jsonl")
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")
        self._seq = 0
        self.events: List[Dict[str, Any]] = []

    def _emit(self, kind: str, payload: Dict[str, Any]) -> None:
        self._seq += 1
        # ts is the monotonic event order, not wall-clock: deterministic and
        # enough to reconstruct sequence for replay.
        event = {"run_id": self.run_id, "seq": self._seq, "kind": kind, **payload}
        self.events.append(event)
        self._fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    # -- event types ----------------------------------------------------

    def run_start(
        self,
        *,
        provider: str,
        model: str,
        task: str,
        system: str,
        tools: Optional[List[ToolSpec]] = None,
    ) -> None:
        # Record the advertised tool specs (not just a count) so a trace is
        # self-describing enough to reconstruct the exact request the model saw -
        # needed by v1 deterministic replay.
        self._emit(
            "run_start",
            {
                "provider": provider,
                "model": model,
                "task": task,
                "system": system,
                "tools": [asdict(t) for t in (tools or [])],
            },
        )

    def llm_request(self, *, step: int, provider: str, model: str, n_messages: int, n_tools: int) -> None:
        self._emit(
            "llm_request",
            {"step": step, "provider": provider, "model": model,
             "n_messages": n_messages, "n_tools": n_tools},
        )

    def llm_response(self, *, step: int, turn: AssistantTurn) -> None:
        self._emit(
            "llm_response",
            {
                "step": step,
                "text": turn.text,
                "stop_reason": turn.stop_reason,
                "tool_calls": [asdict(tc) for tc in turn.tool_calls],
                "usage": asdict(turn.usage),
                "extra_usages": [asdict(u) for u in turn.extra_usages],
            },
        )

    def tool_call(self, *, step: int, call: ToolCall) -> None:
        self._emit("tool_call", {"step": step, **asdict(call)})

    def tool_result(self, *, step: int, result: ToolResult) -> None:
        self._emit("tool_result", {"step": step, **asdict(result)})

    def sandbox(self, *, event: str, kind: str, name: str, detail: str = "") -> None:
        """A sandbox lifecycle event (setup / teardown / error).

        Note ``sandbox_kind`` (not ``kind``) so it can't collide with the event's
        own top-level ``kind`` field in ``_emit``.
        """
        self._emit(
            "sandbox",
            {"event": event, "sandbox_kind": kind, "name": name, "detail": detail},
        )

    def run_end(self, *, status: str, steps: int, final_text: str) -> None:
        self._emit("run_end", {"status": status, "steps": steps, "final_text": final_text})

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "Tracer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
