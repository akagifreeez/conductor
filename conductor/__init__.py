"""Conductor - a vendor-neutral control plane for LLM agents.

A single self-built tool-use loop drives any provider (Claude via the official
Anthropic SDK, any OpenAI-compatible API, local Ollama, or an offline scripted
double) behind one neutral interface, with JSONL tracing and a per-provider cost
ledger reused from token-router.

    from conductor import Orchestrator, ToolRegistry, READONLY_TOOLS
    from conductor.backends.scripted import ScriptedBackend, ScriptedTurn

The network adapters live in ``conductor.backends.anthropic_adapter`` and
``conductor.backends.openai_compat`` and are imported on demand (so the core,
the offline path, and the tests have no hard dependency on the Anthropic SDK).
"""
from .backends.base import (
    AgentBackend,
    AssistantTurn,
    Message,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)
from .backends.scripted import ScriptedBackend, ScriptedTurn
from .orchestrator import DEFAULT_SYSTEM, Orchestrator, RunResult
from .tools import READONLY_TOOLS, SandboxRequiredError, Tool, ToolRegistry
from .tracer import Tracer

__version__ = "0.1.0"

__all__ = [
    "AgentBackend",
    "AssistantTurn",
    "Message",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    "Usage",
    "ScriptedBackend",
    "ScriptedTurn",
    "Orchestrator",
    "RunResult",
    "DEFAULT_SYSTEM",
    "Tool",
    "ToolRegistry",
    "SandboxRequiredError",
    "READONLY_TOOLS",
    "Tracer",
    "__version__",
]
