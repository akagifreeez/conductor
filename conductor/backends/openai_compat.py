"""One adapter for every OpenAI-compatible chat-completions API.

A single ``base_url`` swap points this at OpenAI, OpenRouter, Groq, Mistral,
Together, Fireworks, a local Ollama / LM Studio / vLLM server, or Gemini's
OpenAI-compatible endpoint - they all speak the same ``/chat/completions`` wire
format with the same ``tools`` (function-calling) shape. That is the whole point
of the adapter layer: ~one HTTP client covers most of the market plus local.

It is a thin ``requests`` client wrapped in token-router's proven
``ResilientClient`` (retry / backoff / rate-limit / TTL cache, itself ported
from hl-read), so it inherits well-tested resilience instead of reinventing it.
The job here is purely translation: neutral ``Message``/``ToolSpec`` -> OpenAI
wire shape, and the response's ``tool_calls`` -> neutral ``ToolCall`` + ``Usage``.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import requests
from token_router.models.base import ModelError, Usage, count_tokens
from token_router.models._http import HttpError, ResilientClient

from .base import AgentBackend, AssistantTurn, Message, ToolCall, ToolSpec, estimate_prompt_text


def _default_base_url() -> str:
    explicit = os.environ.get("OPENAI_BASE_URL")
    if explicit:
        return explicit.rstrip("/")
    # Default to a local Ollama server's OpenAI-compatible endpoint so the
    # zero-cost local path works out of the box.
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    if "://" not in host:  # add a scheme only when one is genuinely absent
        host = "http://" + host
    return host.rstrip("/") + "/v1"


class OpenAICompatAdapter(AgentBackend, ResilientClient):
    """An ``AgentBackend`` over any OpenAI-compatible chat-completions endpoint.

    ``backend`` is the cost-ledger provider label. Default ``"openai-compat"``;
    set it to ``"local"`` when pointing at Ollama/LM Studio so the ledger prices
    that traffic at $0 (local inference has no API cost). ``api_key`` falls back
    to ``$OPENAI_API_KEY`` and is omitted entirely for keyless local servers.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        backend: str = "openai-compat",
        max_retries: int = 3,
        backoff_base: float = 0.4,
        backoff_max: float = 8.0,
        rate_limit_per_min: Optional[float] = None,
        cache_ttl: float = 0.0,
        http_timeout: Optional[float] = 120.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        ResilientClient.__init__(
            self,
            max_retries=max_retries,
            backoff_base=backoff_base,
            backoff_max=backoff_max,
            rate_limit_per_min=rate_limit_per_min,
            cache_ttl=cache_ttl,
            http_timeout=http_timeout,
        )
        self.name = model
        self.backend = backend
        self.base_url = (base_url or _default_base_url()).rstrip("/")
        # Empty key is fine for local servers; only sent if present.
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self._session = session or requests.Session()

    # -- translation: neutral -> OpenAI wire shape ----------------------

    @staticmethod
    def _to_openai_messages(system: str, messages: List[Message]) -> List[dict]:
        out: List[dict] = []
        if system:
            out.append({"role": "system", "content": system})
        for m in messages:
            if m.role == "assistant":
                msg: Dict[str, Any] = {"role": "assistant", "content": m.text or None}
                if m.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": tc.call_id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                            },
                        }
                        for tc in m.tool_calls
                    ]
                out.append(msg)
            elif m.tool_results:
                # OpenAI: one role="tool" message per result. The wire format has
                # no first-class error flag, so preserve the neutral is_error
                # signal by prefixing the content (the Anthropic path keeps the
                # structured is_error field) - otherwise the same neutral
                # ToolResult would tell Claude it errored but not an OpenAI model.
                for r in m.tool_results:
                    content = f"[tool error] {r.content}" if r.is_error else r.content
                    out.append(
                        {"role": "tool", "tool_call_id": r.call_id, "content": content}
                    )
            else:
                out.append({"role": m.role, "content": m.text})
        return out

    @staticmethod
    def _to_openai_tools(tools: List[ToolSpec]) -> List[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    # -- wire -----------------------------------------------------------

    def _post_chat(self, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = self._session.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            data=json.dumps(body),
            timeout=self.http_timeout,
        )
        if resp.status_code >= 400:
            snippet = (resp.text or "")[:300]
            raise HttpError(f"HTTP {resp.status_code} from {self.base_url}: {snippet}", resp.status_code)
        return resp.json()

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
        body: Dict[str, Any] = {
            "model": self.name,
            "messages": self._to_openai_messages(system, messages),
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        if tools:
            body["tools"] = self._to_openai_tools(tools)

        t0 = time.monotonic()
        data = self._call(self._post_chat, body)
        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        return self._parse(system, messages, data, latency_ms)

    def _parse(
        self, system: str, messages: List[Message], data: dict, latency_ms: float
    ) -> AssistantTurn:
        try:
            choice = data["choices"][0]
            msg = choice["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise ModelError(f"unexpected OpenAI-compatible response shape: {e}") from e

        text = msg.get("content") or ""
        raw_calls = msg.get("tool_calls") or []
        calls: List[ToolCall] = []
        for tc in raw_calls:
            fn = tc.get("function", {})
            raw_args = fn.get("arguments", "") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except (json.JSONDecodeError, TypeError, ValueError):
                # Don't crash on a malformed arg blob; pass it through so the tool
                # layer reports a clear error the model can recover from.
                args = {"__raw__": raw_args}
            calls.append(
                ToolCall(call_id=tc.get("id", ""), name=fn.get("name", ""), arguments=args)
            )

        stop_reason = "tool_use" if calls else "end"

        u = data.get("usage") or {}
        pt = u.get("prompt_tokens")
        ct = u.get("completion_tokens")
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
