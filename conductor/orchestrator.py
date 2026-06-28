"""The self-built tool-use loop - Conductor's core.

This is the orchestrator the whole project is named for: a single loop that
drives *any* ``AgentBackend`` (Claude, an OpenAI-compatible API, local Ollama, or
the scripted double) through the same cycle:

    user task
      -> backend.step()                      (one assistant turn)
         -> if tool calls: run each via the ToolRegistry, feed results back
         -> else: done, return the final text

It is **not** built on any vendor's agent framework - it is plain Python over
each provider's raw tool-use primitive, which is exactly what makes it
vendor-neutral (see the build plan's "self-built orchestrator" claim). Every LLM
call and every tool call is:

  * written to the JSONL ``Tracer`` (observability + future replay), and
  * recorded in token-router's ``Ledger`` keyed by ``Usage.backend`` (so the
    per-provider cost split is automatic).

The loop also enforces the safety boundary by delegating tool execution to the
``ToolRegistry``, which refuses dangerous tools unless a sandbox is wired in.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from token_router.accounting import Ledger

from .backends.base import AgentBackend, Message
from .pricing import make_pricing
from .tools.registry import ToolRegistry
from .tracer import Tracer

# Router-specific fields token-router's Ledger.summary() emits that are
# meaningless for Conductor (they hard-code "fireworks" as the only remote
# backend). Dropped from the summary Conductor exposes so a consumer is never
# told, e.g., that paid Anthropic traffic was "100% kept local".
_ROUTER_ONLY_SUMMARY_KEYS = ("kept_local_tasks", "escalated_tasks", "local_keep_rate")


def conductor_summary(ledger: Ledger) -> dict:
    """``Ledger.summary()`` projected to the keys meaningful for Conductor.

    Keeps the per-provider ``by_backend`` split (the part Conductor relies on)
    and drops token-router's router-specific local/remote fields.
    """
    s = dict(ledger.summary())
    for k in _ROUTER_ONLY_SUMMARY_KEYS:
        s.pop(k, None)
    return s

DEFAULT_SYSTEM = (
    "You are a helpful assistant with access to tools. Use a tool when it helps "
    "answer the user's request, then give a concise final answer."
)


@dataclass
class RunResult:
    """The outcome of one ``run``."""

    final_text: str
    steps: int
    status: str           # "completed" | "max_steps"
    trace_path: str
    ledger_summary: dict


class Orchestrator:
    """Drives one ``AgentBackend`` through a tool-use loop with full accounting.

    ``max_steps`` bounds the loop so a model that keeps requesting tools can
    never hang the run (it terminates with ``status="max_steps"``). ``run_id``
    tags the trace file and ledger rows.
    """

    def __init__(
        self,
        backend: AgentBackend,
        registry: ToolRegistry,
        *,
        run_id: str,
        system: str = DEFAULT_SYSTEM,
        max_steps: int = 8,
        trace_dir: str = "traces",
        ledger: Optional[Ledger] = None,
        max_tokens: int = 1024,
    ) -> None:
        self.backend = backend
        self.registry = registry
        self.run_id = run_id
        self.system = system
        self.max_steps = max(1, int(max_steps))
        self.trace_dir = trace_dir
        self.max_tokens = max_tokens
        # A fresh ledger per run unless the caller shares one across runs (e.g.
        # to aggregate the per-provider cost split of a two-provider demo). An
        # owned ledger is streamed to its own JSONL (crash-durable, mirroring the
        # trace) and closed by us; a shared/injected ledger is the caller's to
        # configure and close.
        self._owns_ledger = ledger is None
        if ledger is not None:
            self.ledger = ledger
        else:
            ledger_path = os.path.join(trace_dir, f"ledger-{run_id}.jsonl")
            self.ledger = Ledger(pricing=make_pricing(), jsonl_path=ledger_path)

    def run(self, task: str) -> RunResult:
        tools = self.registry.specs()
        messages: List[Message] = [Message(role="user", text=task)]
        final_text = ""
        # "max_steps" means: truncated before the model produced a tool-free final
        # answer. final_text below still carries the last turn's text (if any), so
        # a borderline run isn't returned empty - it's just flagged as truncated.
        status = "max_steps"
        steps = 0

        tracer = Tracer(self.run_id, trace_dir=self.trace_dir)
        try:
            tracer.run_start(
                provider=self.backend.backend,
                model=self.backend.name,
                task=task,
                system=self.system,
                tools=tools,
            )
            try:
                for step in range(self.max_steps):
                    steps = step + 1
                    tracer.llm_request(
                        step=steps,
                        provider=self.backend.backend,
                        model=self.backend.name,
                        n_messages=len(messages),
                        n_tools=len(tools),
                    )
                    turn = self.backend.step(
                        system=self.system,
                        messages=messages,
                        tools=tools,
                        max_tokens=self.max_tokens,
                    )
                    tracer.llm_response(step=steps, turn=turn)
                    # Every LLM call hits the ledger, keyed by provider via Usage.backend.
                    self.ledger.record(self.run_id, stage="llm", usage=turn.usage)

                    # Keep the best-available answer each turn: a turn may carry
                    # text alongside tool calls, and if we hit max_steps on a
                    # tool-requesting turn we still want its narration, not "".
                    if turn.text:
                        final_text = turn.text

                    # Record the assistant turn into history so the next step (and
                    # the provider) see its tool requests.
                    messages.append(turn.as_message())

                    if not turn.tool_calls:
                        final_text = turn.text
                        status = "completed"
                        break

                    # Run each requested tool and feed the results back as one
                    # neutral "tool results" user turn.
                    results = []
                    for call in turn.tool_calls:
                        tracer.tool_call(step=steps, call=call)
                        result = self.registry.execute(call.call_id, call.name, call.arguments)
                        tracer.tool_result(step=steps, result=result)
                        results.append(result)
                    messages.append(Message(role="user", tool_results=results))

                tracer.run_end(status=status, steps=steps, final_text=final_text)
            except Exception as exc:  # noqa: BLE001 - ensure a terminal trace event
                # A backend failure mid-run must still leave a terminated trace,
                # so replay/inspection tooling sees a run_end (status="error")
                # rather than a silently truncated file. Then re-raise.
                tracer.run_end(
                    status="error", steps=steps, final_text=f"{type(exc).__name__}: {exc}"
                )
                raise
        finally:
            tracer.close()
            if self._owns_ledger:
                self.ledger.close()

        return RunResult(
            final_text=final_text,
            steps=steps,
            status=status,
            trace_path=tracer.path,
            ledger_summary=conductor_summary(self.ledger),
        )
