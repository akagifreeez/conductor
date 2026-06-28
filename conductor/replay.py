"""Deterministic replay - reconstruct a run from its recorded trace.

"Deterministic" here means the recorded **I/O is reproduced**, not that the LLM
is re-run bit-for-bit (LLM output isn't deterministic; the trace is). Replay
re-drives the exact same loop, but:

* the ``ReplayBackend`` returns the recorded assistant turns (from ``llm_response``
  events) instead of calling a provider, and
* the ``ReplayRegistry`` returns the recorded tool results (from ``tool_result``
  events) instead of re-executing tools.

So a replay produces a fresh trace that should match the original's tool I/O,
final answer, and per-provider cost. This proves the trace is **complete and
sufficient** to reconstruct the run deterministically - the foundation for audit
and (later) what-if replay. No provider key, no tool side effects, no sandbox.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from token_router.models.base import Usage

from .backends.base import AgentBackend, AssistantTurn, Message, ToolCall, ToolResult, ToolSpec
from .orchestrator import Orchestrator, RunResult
from .tools.registry import ToolRegistry


def load_trace(path: str) -> List[dict]:
    """Read a JSONL trace into a list of event dicts (in recorded order)."""
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _only(d: dict, cls) -> dict:
    """Keep only ``cls``'s dataclass fields, so a forward-added trace field can't
    crash reconstruction with an unexpected-keyword TypeError."""
    from dataclasses import fields

    allowed = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in allowed}


def _turn_from_event(ev: dict) -> AssistantTurn:
    seq = ev.get("seq", "?")
    try:
        tool_calls = [ToolCall(**_only(tc, ToolCall)) for tc in ev.get("tool_calls", [])]
        usage = Usage(**_only(ev["usage"], Usage))
        extra = [Usage(**_only(u, Usage)) for u in ev.get("extra_usages", [])]
    except (KeyError, TypeError) as e:
        # Turn a malformed/incomplete llm_response into a single clear error,
        # consistent with the run_start guard (honest "incomplete trace").
        raise ValueError(f"incomplete/invalid llm_response event at seq={seq}: {e}") from e
    return AssistantTurn(
        text=ev.get("text", ""),
        tool_calls=tool_calls,
        stop_reason=ev.get("stop_reason", "end"),
        usage=usage,
        extra_usages=extra,
    )


class ReplayBackend(AgentBackend):
    """Replays recorded assistant turns in order (no provider call)."""

    def __init__(self, turns: List[AssistantTurn], *, name: str, backend: str) -> None:
        self.name = name
        self.backend = backend
        self._turns = list(turns)
        self._i = 0

    def step(self, *, system, messages, tools, max_tokens=1024, temperature=0.0) -> AssistantTurn:
        if self._i >= len(self._turns):
            # Trace ran out of turns; end cleanly (defensive).
            return AssistantTurn(
                text="", tool_calls=[], stop_reason="end",
                usage=Usage(prompt_tokens=0, completion_tokens=0, model=self.name, backend=self.backend),
            )
        turn = self._turns[self._i]
        self._i += 1
        return turn


class ReplayRegistry(ToolRegistry):
    """Returns recorded tool results instead of executing tools.

    Results are keyed by call_id but stored as a FIFO queue per id, so a trace
    that legitimately reuses a call_id replays each occurrence in recorded order
    rather than silently serving the last one to every call.
    """

    def __init__(self, specs: List[ToolSpec], results: Dict[str, List[ToolResult]]) -> None:
        super().__init__()
        self._specs = list(specs)
        self._results = {k: list(v) for k, v in results.items()}

    def specs(self) -> List[ToolSpec]:
        return list(self._specs)

    def execute(self, call_id: str, name: str, arguments: dict) -> ToolResult:
        queue = self._results.get(call_id)
        if queue:
            return queue.pop(0)
        return ToolResult(
            call_id=call_id, name=name,
            content=f"Error: no recorded result for call_id {call_id!r}", is_error=True,
        )


def replay_trace(path: str, *, run_id: str, trace_dir: str = "traces") -> Tuple[RunResult, dict]:
    """Replay the trace at ``path``; return the new RunResult and a comparison.

    The comparison reports whether the replayed run reproduced the original's
    ordered tool-result contents and final answer.
    """
    events = load_trace(path)
    if not events or events[0].get("kind") != "run_start":
        raise ValueError("not a valid run trace (missing run_start)")
    start = events[0]
    task = start.get("task", "")
    system = start.get("system", "")
    specs = [ToolSpec(**t) for t in start.get("tools", [])]

    turns = [_turn_from_event(e) for e in events if e.get("kind") == "llm_response"]
    # FIFO queue per call_id so duplicate ids replay in recorded order.
    results: Dict[str, List[ToolResult]] = {}
    for e in events:
        if e.get("kind") != "tool_result":
            continue
        if "call_id" not in e:
            raise ValueError(f"incomplete tool_result event at seq={e.get('seq', '?')}: no call_id")
        results.setdefault(e["call_id"], []).append(
            ToolResult(
                call_id=e["call_id"], name=e.get("name", ""),
                content=e.get("content", ""), is_error=e.get("is_error", False),
            )
        )

    backend = ReplayBackend(turns, name=start.get("model", "replay"), backend=start.get("provider", "replay"))
    registry = ReplayRegistry(specs, results)
    orch = Orchestrator(
        backend, registry, run_id=run_id, system=system,
        max_steps=max(len(turns), 1), trace_dir=trace_dir,
    )
    res = orch.run(task)

    original_results = [e.get("content", "") for e in events if e.get("kind") == "tool_result"]
    original_final = next((e.get("final_text", "") for e in events if e.get("kind") == "run_end"), "")
    new_events = load_trace(res.trace_path)
    new_results = [e.get("content", "") for e in new_events if e.get("kind") == "tool_result"]

    tool_results_match = new_results == original_results
    final_match = res.final_text == original_final
    comparison = {
        "match": tool_results_match and final_match,
        "tool_results_match": tool_results_match,
        "final_match": final_match,
        "original_final": original_final,
        "replayed_final": res.final_text,
        "n_tool_results": len(original_results),
        "n_tool_results_replayed": len(new_results),  # may differ on a truncated trace
    }
    return res, comparison
