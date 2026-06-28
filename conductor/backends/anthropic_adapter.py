"""Claude as one backend among many - via the official ``anthropic`` SDK.

Per the project's vendor-neutral stance, Claude is a first-class backend, not
the only one, and it is reached through Anthropic's *official* SDK (not an
OpenAI-compatible shim). This adapter's only job is translation between the
neutral types and Anthropic's Messages API:

* neutral ``Message`` -> Anthropic ``messages`` (system is a top-level param;
  assistant tool requests are ``tool_use`` content blocks; tool outputs go back
  as ``tool_result`` blocks inside a user message).
* Anthropic response content blocks -> neutral ``text`` + ``ToolCall`` list.
* ``response.usage`` -> token-router ``Usage`` (so the same ledger prices it).

The ``anthropic`` package is imported lazily inside ``__init__`` so that the rest
of Conductor - the loop, tracer, ledger, OpenAI-compat path, and all offline
tests - has zero hard dependency on it. Install it with the ``[anthropic]``
extra only if you actually use this backend.

Design note (intentional): extended thinking is **not** enabled here. Thinking
blocks are Anthropic-specific and must be replayed verbatim, which has no
cross-provider analogue; enabling it would leak a provider detail into the
neutral conversation history and break the abstraction. A vendor-neutral loop
deliberately stays on the plain tool-use path.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from token_router.models.base import ModelError, Usage, count_tokens

from .base import AgentBackend, AssistantTurn, Message, ToolCall, ToolSpec, estimate_prompt_text

DEFAULT_MODEL = "claude-opus-4-8"


class AnthropicAdapter(AgentBackend):
    """An ``AgentBackend`` over Anthropic's Messages API (official SDK)."""

    backend = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 120.0,
        client: Optional[Any] = None,
    ) -> None:
        self.name = model
        if client is not None:
            # Injectable for tests; avoids importing the SDK at all offline.
            self._client = client
        else:
            try:
                import anthropic  # lazy: only needed when this backend is used
            except ImportError as e:  # pragma: no cover - environment dependent
                raise ModelError(
                    "the 'anthropic' package is required for AnthropicAdapter; "
                    "install it with: pip install 'conductor-cp[anthropic]'"
                ) from e
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ModelError("ANTHROPIC_API_KEY is not set (no api_key provided)")
            self._client = anthropic.Anthropic(
                api_key=key, max_retries=max_retries, timeout=timeout
            )

    # -- translation: neutral -> Anthropic wire shape -------------------

    @staticmethod
    def _to_anthropic_messages(messages: List[Message]) -> List[dict]:
        out: List[dict] = []
        for m in messages:
            if m.role == "assistant":
                content: List[dict] = []
                if m.text:
                    content.append({"type": "text", "text": m.text})
                for tc in m.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.call_id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                # The Messages API rejects a message with an empty content list
                # (400). An assistant turn with neither text nor tool calls carries
                # no information, so skip it - the OpenAI path tolerates content:null
                # for the same neutral message, and dropping a no-op turn keeps the
                # two providers' validity symmetric.
                if not content:
                    continue
                out.append({"role": "assistant", "content": content})
            elif m.tool_results:
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": r.call_id,
                                "content": r.content,
                                "is_error": r.is_error,
                            }
                            for r in m.tool_results
                        ],
                    }
                )
            else:
                out.append({"role": m.role, "content": m.text})
        return out

    @staticmethod
    def _to_anthropic_tools(tools: List[ToolSpec]) -> List[dict]:
        return [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in tools
        ]

    # -- AgentBackend ---------------------------------------------------

    def step(
        self,
        *,
        system: str,
        messages: List[Message],
        tools: List[ToolSpec],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> AssistantTurn:
        kwargs: Dict[str, Any] = {
            "model": self.name,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "messages": self._to_anthropic_messages(messages),
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._to_anthropic_tools(tools)

        t0 = time.monotonic()
        try:
            resp = self._client.messages.create(**kwargs)
        except Exception as e:  # noqa: BLE001 - normalize to ModelError for the loop
            raise ModelError(f"Anthropic request failed: {type(e).__name__}: {e}") from e
        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        return self._parse(system, messages, resp, latency_ms)

    def _parse(
        self, system: str, messages: List[Message], resp: Any, latency_ms: float
    ) -> AssistantTurn:
        text_parts: List[str] = []
        calls: List[ToolCall] = []
        for block in getattr(resp, "content", []) or []:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(getattr(block, "text", "") or "")
            elif btype == "tool_use":
                calls.append(
                    ToolCall(
                        call_id=getattr(block, "id", ""),
                        name=getattr(block, "name", ""),
                        arguments=dict(getattr(block, "input", {}) or {}),
                    )
                )
        text = "".join(text_parts)
        stop_reason = "tool_use" if calls else "end"

        usage_obj = getattr(resp, "usage", None)
        pt = getattr(usage_obj, "input_tokens", None) if usage_obj else None
        ct = getattr(usage_obj, "output_tokens", None) if usage_obj else None
        estimated = pt is None or ct is None
        prompt_text = estimate_prompt_text(system, messages)
        usage = Usage(
            prompt_tokens=int(pt) if pt is not None else count_tokens(prompt_text, self.name),
            completion_tokens=int(ct) if ct is not None else count_tokens(text, self.name),
            model=self.name,
            backend=self.backend,
            latency_ms=latency_ms,
            estimated=estimated,
        )
        return AssistantTurn(text=text, tool_calls=calls, stop_reason=stop_reason, usage=usage)
