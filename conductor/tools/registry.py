"""Define-once tools, plus the registry that runs them.

A ``Tool`` is declared a single time with a provider-agnostic ``ToolSpec`` (the
schema the model sees) and a Python ``handler`` (what actually runs). Every
adapter converts the *same* ``ToolSpec`` into its provider's tool-declaration
format, so adding a provider never means re-declaring tools.

The registry is also where Conductor's safety boundary lives. Each tool carries
a ``dangerous`` flag; ``ToolRegistry.execute`` refuses to run a dangerous tool
unless a sandbox executor is wired in. In v0 there is no sandbox, so dangerous
tools are declared-but-blocked - the gate is real from day one, the Proxmox
executor that satisfies it lands in v1 (see the build plan). This mirrors
agent-docsmith's allowlist idea: the boundary is enforced in the loop, not left
to the model's goodwill.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..backends.base import ToolResult, ToolSpec

# A handler takes the parsed argument dict and returns any JSON-serializable
# value; the registry serializes the return to text before it re-enters the
# conversation.
ToolHandler = Callable[[Dict[str, Any]], Any]


@dataclass(frozen=True)
class Tool:
    """A declared tool: the schema the model sees + the code that runs it.

    ``dangerous=True`` marks a tool whose effects must be isolated (shell, file
    writes, network mutation). The registry will not execute a dangerous tool
    without a sandbox executor - see ``ToolRegistry.execute``.
    """

    spec: ToolSpec
    handler: ToolHandler
    dangerous: bool = False


class SandboxRequiredError(RuntimeError):
    """Raised when a dangerous tool is invoked but no sandbox is wired in.

    This is the v0 manifestation of the "dangerous tools must be sandboxed"
    rule: the gate exists and fires; the Proxmox executor that would satisfy it
    is a v1 deliverable.
    """


class ToolRegistry:
    """Holds the declared tools and runs them, enforcing the sandbox gate.

    ``sandbox`` (optional) is any callable ``(Tool, args) -> Any``; when present,
    dangerous tools are dispatched THROUGH it instead of being run in-process.
    Left ``None`` (the v0 default), dangerous tools are blocked.
    """

    def __init__(self, sandbox: Optional[Callable[[Tool, Dict[str, Any]], Any]] = None) -> None:
        self._tools: Dict[str, Tool] = {}
        self._sandbox = sandbox

    def set_sandbox(self, sandbox: Optional[Callable[[Tool, Dict[str, Any]], Any]]) -> None:
        """Wire (or clear) the sandbox executor dangerous tools dispatch through."""
        self._sandbox = sandbox

    def register(self, tool: Tool) -> None:
        if tool.spec.name in self._tools:
            raise ValueError(f"tool already registered: {tool.spec.name}")
        self._tools[tool.spec.name] = tool

    def specs(self) -> List[ToolSpec]:
        """The provider-agnostic specs to advertise to a backend."""
        return [t.spec for t in self._tools.values()]

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def execute(self, call_id: str, name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Run tool ``name`` with ``arguments``; never raise into the loop.

        Any failure (unknown tool, blocked-by-sandbox, handler exception) is
        returned as an *error* ``ToolResult`` so the model can see it and try to
        recover - the orchestration loop stays alive. ``call_id`` is echoed back
        so the provider can pair this result with its request.
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                call_id=call_id,
                name=name,
                content=f"Error: unknown tool {name!r}.",
                is_error=True,
            )

        try:
            if tool.dangerous:
                if self._sandbox is None:
                    raise SandboxRequiredError(
                        f"tool {name!r} is marked dangerous and requires a sandbox "
                        f"executor, which is not configured (v0). Refusing to run it "
                        f"in-process."
                    )
                value = self._sandbox(tool, arguments)
            else:
                value = tool.handler(arguments)
        except SandboxRequiredError as e:
            return ToolResult(call_id=call_id, name=name, content=f"Blocked: {e}", is_error=True)
        except Exception as e:  # noqa: BLE001 - surfaced to the model, not crashed
            return ToolResult(
                call_id=call_id,
                name=name,
                content=f"Error running {name!r}: {type(e).__name__}: {e}",
                is_error=True,
            )

        return ToolResult(call_id=call_id, name=name, content=_to_text(value))


def _to_text(value: Any) -> str:
    """Serialize a handler return to the string the model will read."""
    if isinstance(value, str):
        return value
    import json

    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)
