"""CascadeBackend - cheap-first, escalate-on-low-confidence cost routing.

This brings token-router's core idea (try the cheap model, only pay for the
strong one when a confidence signal says the cheap answer is unreliable) into the
tool-use loop. It is itself an ``AgentBackend``, so the orchestrator drives it
exactly like any single provider - the cascade is invisible above this layer.

Per step:
  1. call the CHEAP backend (e.g. local Ollama),
  2. ask a confidence ``gate`` whether that turn is good enough,
  3. if yes -> return it; if no -> call the STRONG backend (e.g. Claude) and
     return that turn, attaching the cheap attempt's ``Usage`` as ``extra_usages``
     so the loop's ledger records BOTH costs (escalation is never hidden).

Because each underlying turn carries its own provider's ``Usage.backend``, the
ledger's per-provider split shows exactly how much went to local vs remote.

Cost contract: for the cheap leg to be costed at $0, the cheap backend must use
``Usage.backend == "local"`` (e.g. ``OpenAICompatAdapter(..., backend="local")``
pointed at Ollama). A cheap backend with any other label is priced by the model
table / placeholder rate, not free - the $0 is a property of the "local" label,
not of "being the cheap leg".

Honesty: the default gate is a small **heuristic** (did the cheap model make
progress / give a non-trivial, non-hedging answer?), NOT a learned router. Pass
your own ``gate`` to change the policy.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, List, Optional

from .base import AgentBackend, AssistantTurn, Message, ToolSpec

# A gate decides whether the cheap turn is acceptable (True = keep, don't escalate).
ConfidenceGate = Callable[[AssistantTurn], bool]

_HEDGE_MARKERS = (
    "i don't know", "i do not know", "not sure", "cannot help", "can't help",
    "unable to", "as an ai", "i'm not able", "no idea",
)


def default_gate(turn: AssistantTurn, *, min_chars: int = 12) -> bool:
    """Heuristic: keep the cheap turn if it's making progress or gave a real answer.

    Acceptable if the cheap model requested a tool (progress) OR produced a
    non-trivial answer with no obvious hedging. Otherwise escalate.

    Known limitation (by design, hence the ``gate=`` escape hatch): ANY tool-call
    turn is treated as progress and kept, even a wrong or malformed tool call -
    this gate does not validate the tool name or arguments. Pass a stricter gate
    (e.g. one that checks the call against the registry's specs) if that matters.
    """
    if turn.tool_calls:
        return True
    text = (turn.text or "").strip()
    if len(text) < min_chars:
        return False
    low = text.lower()
    return not any(m in low for m in _HEDGE_MARKERS)


class CascadeBackend(AgentBackend):
    """A meta-backend that routes each step cheap-first with escalation."""

    def __init__(
        self,
        cheap: AgentBackend,
        strong: AgentBackend,
        *,
        gate: Optional[ConfidenceGate] = None,
        name: Optional[str] = None,
    ) -> None:
        self.cheap = cheap
        self.strong = strong
        self.gate = gate or default_gate
        self.name = name or f"cascade({cheap.name}->{strong.name})"
        # Label is informational; the real per-call provider labels live on each
        # underlying turn's Usage.backend (so the ledger splits cheap vs strong).
        self.backend = "cascade"

    def step(
        self,
        *,
        system: str,
        messages: List[Message],
        tools: List[ToolSpec],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> AssistantTurn:
        cheap_turn = self.cheap.step(
            system=system, messages=messages, tools=tools,
            max_tokens=max_tokens, temperature=temperature,
        )
        if self.gate(cheap_turn):
            return cheap_turn

        # Escalate: pay for the strong model, but carry the cheap attempt's cost.
        strong_turn = self.strong.step(
            system=system, messages=messages, tools=tools,
            max_tokens=max_tokens, temperature=temperature,
        )
        # Return a NEW turn rather than mutating the strong backend's object: a
        # backend that reuses/returns a shared AssistantTurn would otherwise
        # accumulate extra_usages across steps and double-count the cheap leg.
        return replace(
            strong_turn, extra_usages=list(strong_turn.extra_usages) + [cheap_turn.usage]
        )
