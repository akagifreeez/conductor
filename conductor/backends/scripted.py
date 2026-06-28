"""An offline, deterministic backend - the test double for the whole stack.

This is the direct analogue of token-router's ``ScriptedModel``: it lets the
entire tool-use loop, the JSONL tracer, and the cost ledger be exercised end to
end with **no network, no API key, and no local model server** - so behavior is
reproducible in CI and the v0 goal ("the same task run through two providers,
all I/O traced, cost split per provider") can be demonstrated honestly offline.

It is a genuine ``AgentBackend``: it returns real ``AssistantTurn`` objects with
populated ``Usage`` (token counts come from token-router's ``count_tokens``
estimate, flagged ``estimated=True``), so the ledger numbers it produces are
computed the same way as a real backend's - they are clearly labeled estimates,
not fabricated figures.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from token_router.models.base import Usage, count_tokens

from .base import AgentBackend, AssistantTurn, Message, ToolCall, ToolSpec, estimate_prompt_text


@dataclass
class ScriptedTurn:
    """One pre-programmed assistant turn.

    Provide ``tool_calls`` to make the model "request" tools (the loop runs them
    and calls the backend again for the next turn), or ``text`` for a final
    answer. A turn may carry both: some text plus tool calls.
    """

    text: str = ""
    # each entry is (tool_name, arguments)
    tool_calls: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list)


class ScriptedBackend(AgentBackend):
    """Replays a fixed list of ``ScriptedTurn`` across successive ``step()`` calls.

    ``backend`` is the provider label that flows into the ledger's per-provider
    split - set it to e.g. ``"anthropic"`` or ``"local"`` to simulate distinct
    providers offline. If ``step()`` is called more times than there are scripted
    turns, it returns a terminal empty turn (defensive: the loop must still
    converge rather than hang).
    """

    def __init__(
        self,
        turns: List[ScriptedTurn],
        *,
        name: str = "scripted-1",
        backend: str = "scripted",
        latency_ms: float = 0.0,
    ) -> None:
        self.name = name
        self.backend = backend
        self._turns = list(turns)
        self._i = 0
        self._latency_ms = float(latency_ms)

    def step(
        self,
        *,
        system: str,
        messages: List[Message],
        tools: List[ToolSpec],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> AssistantTurn:
        t0 = time.monotonic()
        if self._i >= len(self._turns):
            scripted = ScriptedTurn(text="")  # terminal: nothing left to say
        else:
            scripted = self._turns[self._i]
        idx = self._i
        self._i += 1

        calls: List[ToolCall] = [
            ToolCall(call_id=f"{self.backend}-call-{idx}-{j}", name=tn, arguments=dict(ta))
            for j, (tn, ta) in enumerate(scripted.tool_calls)
        ]
        stop_reason = "tool_use" if calls else "end"

        # Estimate tokens the same way token-router does when a backend doesn't
        # report usage. The scripted double is always estimated; fold in tool I/O
        # so the estimate reflects real conversation weight (shared helper).
        prompt_text = estimate_prompt_text(system, messages)
        usage = Usage(
            prompt_tokens=count_tokens(prompt_text, self.name),
            completion_tokens=count_tokens(scripted.text, self.name),
            model=self.name,
            backend=self.backend,
            latency_ms=round((time.monotonic() - t0) * 1000 + self._latency_ms, 1),
            estimated=True,
        )
        return AssistantTurn(
            text=scripted.text,
            tool_calls=calls,
            stop_reason=stop_reason,
            usage=usage,
        )
