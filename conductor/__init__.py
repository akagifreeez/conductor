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
from .backends.cascade import CascadeBackend, default_gate
from .orchestrator import (
    DEFAULT_SYSTEM,
    Orchestrator,
    RunResult,
    conductor_summary,
    ledger_cost_usd,
)
from .coordinator import Coordinator, CoordinatorResult, Job, JobOutcome
from .replay import replay_trace
from .sandbox import (
    ExecResult,
    ProxmoxSandbox,
    Sandbox,
    SandboxExecutor,
    SubprocessSandbox,
    sandbox_selfcheck,
)
from .tools import (
    READONLY_TOOLS,
    SANDBOX_TOOLS,
    SandboxRequiredError,
    Tool,
    ToolRegistry,
)
from .tracer import Tracer

__version__ = "0.3.0"

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
    "CascadeBackend",
    "default_gate",
    "Orchestrator",
    "RunResult",
    "conductor_summary",
    "ledger_cost_usd",
    "DEFAULT_SYSTEM",
    "Coordinator",
    "CoordinatorResult",
    "Job",
    "JobOutcome",
    "replay_trace",
    "Sandbox",
    "SandboxExecutor",
    "SubprocessSandbox",
    "ProxmoxSandbox",
    "sandbox_selfcheck",
    "ExecResult",
    "Tool",
    "ToolRegistry",
    "SandboxRequiredError",
    "READONLY_TOOLS",
    "SANDBOX_TOOLS",
    "Tracer",
    "__version__",
]
