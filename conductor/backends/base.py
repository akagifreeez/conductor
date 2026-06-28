"""The provider-agnostic agent interface every LLM backend honors.

This is the heart of Conductor's vendor-neutrality. The orchestrator (the
self-built tool-use loop) depends ONLY on the types in this module -
``AgentBackend``, ``Message``, ``ToolSpec``, ``ToolCall`` - never on a concrete
provider. Anthropic, any OpenAI-compatible API, and a scripted offline test
double are therefore interchangeable.

It is a deliberate extension of token-router's ``Model``/``Usage`` contract
(``token_router.models.base``): that interface proved one cheap-vs-strong split
behind a single ``complete()`` call; this one keeps the same ``Usage`` accounting
(so the cost ledger is reused verbatim) but upgrades the unit of work from a
one-shot completion to *one assistant turn that may request tool calls* - the
minimum needed to drive a real tool-use loop across providers.

What stays provider-agnostic here, and what each adapter must convert:

* ``Message``  - one conversation turn (user / assistant), neutral shape.
* ``ToolSpec`` - a tool declared ONCE (name + description + JSON-Schema params).
* ``ToolCall`` / ``ToolResult`` - a normalized request to run a tool and its
  outcome, with a stable ``call_id`` linking them across the round-trip.
* ``AssistantTurn`` - what a backend returns: assistant text, any tool calls it
  wants run, why it stopped, and a populated ``Usage`` for the ledger.

Each adapter's only job is to translate this neutral shape <-> its provider's
wire format (Claude ``tool_use``/``tool_result`` blocks; OpenAI
``tool_calls``/``role:"tool"`` messages) and to fill ``Usage``. Nothing above the
adapter ever sees a provider detail.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Reuse token-router's accounting primitive verbatim: every backend reports the
# same Usage, so the existing JSONL Ledger records Conductor runs unchanged and
# the per-provider cost split falls out of Usage.backend for free.
from token_router.models.base import Usage  # noqa: F401  (re-exported for callers)


@dataclass(frozen=True)
class ToolSpec:
    """A tool declared once, provider-agnostically.

    ``parameters`` is a JSON Schema object (the lowest common denominator both
    Anthropic and OpenAI accept). Adapters convert this into the provider's tool
    declaration shape; tool authors never write provider-specific schemas.
    """

    name: str
    description: str
    parameters: Dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """A normalized request from the model to run one tool.

    ``call_id`` is the provider's id for this call (Anthropic ``tool_use.id`` /
    OpenAI ``tool_calls[].id``); it MUST be echoed back on the matching
    ``ToolResult`` so the provider can pair request and result.
    """

    call_id: str
    name: str
    arguments: Dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """The outcome of running one ``ToolCall``.

    ``is_error`` lets the loop hand a failure back to the model (so it can
    recover) instead of crashing the run. ``content`` is always a string - tool
    outputs are serialized to text before they re-enter the conversation.
    """

    call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    """One neutral conversation turn.

    A turn is exactly one of three shapes, mapped to each provider by the adapter:

    * user prompt           -> ``role="user"``, ``text`` set.
    * assistant turn        -> ``role="assistant"``, ``text`` and/or ``tool_calls``.
    * tool results to model -> ``role="user"``, ``tool_results`` set (Anthropic
      packs these as ``tool_result`` blocks in a user message; OpenAI emits one
      ``role="tool"`` message per result - the adapter handles the difference).
    """

    role: str
    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)


@dataclass
class AssistantTurn:
    """What a backend returns for one ``step()``.

    ``stop_reason`` is normalized to ``"tool_use"`` (the model wants tools run)
    or ``"end"`` (the model is done). ``usage`` is always populated so the cost
    ledger can never silently drift - same guarantee as token-router's ``Model``.

    ``extra_usages`` carries the cost of any *discarded* internal calls a
    meta-backend made on the way to this turn (e.g. a cost cascade that tried a
    cheap model first, then escalated). The loop records these in the ledger too,
    so an escalation's full cost is never hidden.
    """

    text: str
    tool_calls: List[ToolCall]
    stop_reason: str
    usage: Usage
    extra_usages: List[Usage] = field(default_factory=list)

    def as_message(self) -> Message:
        """Re-enter this turn into the conversation history as an assistant message."""
        return Message(role="assistant", text=self.text, tool_calls=list(self.tool_calls))


def estimate_prompt_text(system: str, messages: List[Message]) -> str:
    """Concatenate all *billable* text in a conversation, for a token estimate.

    Used ONLY as the fallback when a provider does not report real usage. It
    folds in not just ``Message.text`` but also tool-call names + arguments and
    tool-result content, because those serialize to real tokens on the wire. A
    naive ``"".join(m.text ...)`` would contribute zero for assistant tool-call
    turns (text often empty) and tool-result turns (text always empty), so a
    tool-heavy conversation would be systematically under-counted - defeating
    token-router's ``count_tokens`` guarantee of never under-counting. Shared by
    all backends so the estimate can't drift between them.
    """
    import json

    parts: List[str] = [system]
    for m in messages:
        if m.text:
            parts.append(m.text)
        for tc in m.tool_calls:
            parts.append(tc.name)
            parts.append(json.dumps(tc.arguments, ensure_ascii=False, default=str))
        for r in m.tool_results:
            parts.append(r.content)
    return "".join(parts)


class AgentBackend(abc.ABC):
    """A provider behind a single ``step()`` call.

    Implementations: ``AnthropicAdapter`` (official ``anthropic`` SDK),
    ``OpenAICompatAdapter`` (any OpenAI-compatible REST endpoint, incl. local
    Ollama / LM Studio / vLLM), and ``ScriptedBackend`` (offline test double).

    ``backend`` is the provider label that flows into ``Usage.backend`` and thus
    becomes the key for the per-provider cost split in the ledger
    (e.g. ``"anthropic"``, ``"openai-compat"``, ``"local"``, ``"scripted"``).
    ``name`` is the concrete model id used for pricing.
    """

    name: str
    backend: str

    @abc.abstractmethod
    def step(
        self,
        *,
        system: str,
        messages: List[Message],
        tools: List[ToolSpec],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> AssistantTurn:
        """Produce one assistant turn given the conversation and available tools.

        Must return a populated ``Usage`` and raise ``token_router``'s
        ``ModelError`` (or a subclass) on unrecoverable failure.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Optional cleanup hook (default no-op)."""
        return None
